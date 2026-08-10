"""
Promote a registered model's `dev` alias to `prod`, gated on a metric comparison.

Only moves `prod` to the `dev` candidate when its metric is at least as good as
the model currently serving `prod` (or unconditionally on the very first
promotion, when `prod` does not exist yet). Safe to re-run: if `dev` and `prod`
already point at the same version, nothing happens.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import mlflow
from dotenv import dotenv_values
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


_env = dotenv_values()

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

_minio_user = _env.get("MINIO_ROOT_USER")
_minio_password = _env.get("MINIO_ROOT_PASSWORD")
if _minio_user and _minio_password:
    os.environ["AWS_ACCESS_KEY_ID"] = _minio_user
    os.environ["AWS_SECRET_ACCESS_KEY"] = _minio_password
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")

MODEL_NAMES = [
    "pk_model",
    "composite_pk_model",
    "fk_model",
    "composite_fk_model",
    "denormalization_model",
    "subject_area_model",
]
DEFAULT_METRIC = "test_f1_score"


@dataclass
class PromotionOutcome:
    model_name: str
    action: str
    message: str


def get_alias_version(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def get_metric(
    client: MlflowClient,
    model_version: ModelVersion,
    metric_name: str,
) -> float | None:
    run = client.get_run(model_version.run_id)
    return run.data.metrics.get(metric_name)


def decide_and_promote(
    client: MlflowClient,
    model_name: str,
    metric_name: str = DEFAULT_METRIC,
    dry_run: bool = False,
) -> PromotionOutcome:
    dev_version = get_alias_version(client, model_name, "dev")
    if dev_version is None:
        return PromotionOutcome(
            model_name,
            "no_dev",
            f"{model_name}: no @dev version found, nothing to promote.",
        )

    dev_metric = get_metric(client, dev_version, metric_name)
    if dev_metric is None:
        return PromotionOutcome(
            model_name,
            "missing_metric",
            f"{model_name}: dev version {dev_version.version} "
            f"(run {dev_version.run_id}) has no metric '{metric_name}'.",
        )

    prod_version = get_alias_version(client, model_name, "prod")

    if prod_version is None:
        if not dry_run:
            client.set_registered_model_alias(model_name, "prod", dev_version.version)
        return PromotionOutcome(
            model_name,
            "bootstrap",
            f"{model_name}: first promotion, prod bootstrapped at dev "
            f"v{dev_version.version} ({metric_name}={dev_metric:.4f}).",
        )

    if dev_version.version == prod_version.version:
        return PromotionOutcome(
            model_name,
            "noop_same_version",
            f"{model_name}: prod already at v{prod_version.version} (== dev), nothing to do.",
        )

    prod_metric = get_metric(client, prod_version, metric_name)
    if prod_metric is None:
        return PromotionOutcome(
            model_name,
            "missing_metric",
            f"{model_name}: prod version {prod_version.version} "
            f"(run {prod_version.run_id}) has no metric '{metric_name}'.",
        )

    if dev_metric >= prod_metric:
        if not dry_run:
            client.set_registered_model_alias(model_name, "prod", dev_version.version)
        action = "promoted"
        message = (
            f"{model_name}: promoted dev v{dev_version.version} "
            f"({metric_name}={dev_metric:.4f}) over prod v{prod_version.version} "
            f"({metric_name}={prod_metric:.4f})."
        )
    else:
        action = "rejected"
        message = (
            f"{model_name}: dev v{dev_version.version} ({metric_name}={dev_metric:.4f}) "
            f"< prod v{prod_version.version} ({metric_name}={prod_metric:.4f}); "
            "prod unchanged."
        )

    return PromotionOutcome(model_name, action, message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a model's dev alias to prod, gated on a metric comparison.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=MODEL_NAMES,
        help="Model name to promote (repeatable). Defaults to all models.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric to compare (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the decision without moving any alias.",
    )
    args = parser.parse_args()

    models = args.models or MODEL_NAMES
    client = MlflowClient()

    outcomes = [
        decide_and_promote(client, model_name, args.metric, args.dry_run) for model_name in models
    ]

    for outcome in outcomes:
        print(outcome.message)

    if any(outcome.action == "promoted" for outcome in outcomes) and not args.dry_run:
        print("\nRestart model-service to serve the newly promoted model(s):")
        print("  docker compose restart model-service")

    if any(outcome.action == "missing_metric" for outcome in outcomes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
