"""
Real-world evaluation gate for the normal-form model: the Willibald period-1 tables.

Why this exists (2026-08-07): the model scored 0.97 cross-validated table accuracy on
generated data and got 6 of the 10 real Willibald tables wrong - at 0.97+ confidence -
because the decisive dependencies were shapes the extractor could not see and the
generator never produced. Synthetic CV alone cannot catch that failure class; a small,
hand-verified real-world set can, so it becomes a gate next to the CV baseline.

Two halves:

* ``--extract`` profiles the live ``iceberg.new_predict_data`` tables through
  ``nf_features.build_features`` (the same code path serving uses) and freezes the
  result to ``data/willibald_eval_features.csv``. Re-run it whenever the feature set
  changes - the training script needs the snapshot to carry every FEATURE_COLUMN.
* ``evaluate_on_willibald`` scores a fitted classifier against
  ``data/willibald_ground_truth.csv`` exactly the way the pipeline serves: per-column
  probabilities, confidence-weighted soft vote per table, argmax.

The snapshot is EVALUATION data only. It must never be appended to nf_training.csv -
these ten tables are the closest thing this project has to a held-out production set,
and training on them would turn the gate into a mirror.

Usage:
    python task_3/nf_willibald_eval.py --extract          # refresh the feature snapshot
    python task_3/nf_willibald_eval.py                    # score models:/denormalization_model@dev
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import urllib3


if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "prefect"))

from nf_features import (  # noqa: E402 - path has to be set before this import
    COLUMN_TYPE_DUMMIES,
    FEATURE_COLUMNS,
    build_features,
    encode_column_type,
)


urllib3.disable_warnings()

FEATURES_PATH = REPO_ROOT / "data" / "willibald_eval_features.csv"
GROUND_TRUTH_PATH = REPO_ROOT / "data" / "willibald_ground_truth.csv"

EVAL_SCHEMA = "new_predict_data"


def extract_snapshot(batch_size: int = 20) -> pd.DataFrame:
    """Profile every ground-truth table from Iceberg and freeze the feature rows."""
    from nf_generator import get_trino_engine  # noqa: PLC0415 - Trino only needed here

    truth = pd.read_csv(GROUND_TRUTH_PATH)
    engine = get_trino_engine()
    frames = []
    with engine.connect() as conn:
        for table in truth["table_name"]:
            frames.append(
                build_features(conn, "iceberg", EVAL_SCHEMA, table, batch_size=batch_size),
            )
            print(f"  profiled {EVAL_SCHEMA}.{table}")
    snapshot = pd.concat(frames, ignore_index=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(FEATURES_PATH, index=False)
    print(f"{len(snapshot)} feature rows for {truth.shape[0]} tables -> {FEATURES_PATH}")
    return snapshot


def evaluate_on_willibald(model: ClassifierMixin) -> dict | None:
    """
    Table-level accuracy of ``model`` on the Willibald ground truth, or None when the
    snapshot is missing or stale (its columns no longer cover the feature contract).

    Mirrors the serving path: per-column ``predict_proba``, then a confidence-weighted
    soft vote per table - the mean of the per-column class distributions, argmax.
    """
    if not FEATURES_PATH.exists() or not GROUND_TRUTH_PATH.exists():
        print(
            f"willibald eval skipped: missing {FEATURES_PATH.name} or "
            f"{GROUND_TRUTH_PATH.name} - run nf_willibald_eval.py --extract",
        )
        return None

    features = pd.read_csv(FEATURES_PATH)
    feature_names = [*FEATURE_COLUMNS, *COLUMN_TYPE_DUMMIES]
    encoded = encode_column_type(features)
    missing = [name for name in feature_names if name not in encoded.columns]
    if missing:
        print(
            f"willibald eval skipped: snapshot is missing {missing} - "
            "re-run nf_willibald_eval.py --extract against the current extractor",
        )
        return None

    truth = pd.read_csv(GROUND_TRUTH_PATH).set_index("table_name")["normal_form"]
    X = encoded[feature_names].astype(float)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    per_table: dict[str, dict] = {}
    frame = pd.DataFrame(probabilities, columns=classes)
    frame["table_name"] = encoded["table_name"].to_numpy()
    for table, group in frame.groupby("table_name"):
        distribution = group[classes].mean()
        predicted = int(distribution.idxmax())
        per_table[str(table)] = {
            "predicted": predicted,
            "true": int(truth[table]) if table in truth.index else None,
            "confidence": round(float(distribution.max()), 4),
        }

    scored = {name: row for name, row in per_table.items() if row["true"] is not None}
    hits = sum(1 for row in scored.values() if row["predicted"] == row["true"])
    return {
        "willibald_table_accuracy": round(hits / len(scored), 4) if scored else 0.0,
        "willibald_tables": len(scored),
        "per_table": per_table,
    }


def print_report(result: dict) -> None:
    print(
        f"\nWillibald table accuracy: {result['willibald_table_accuracy']:.4f} "
        f"({result['willibald_tables']} tables)",
    )
    for table, row in sorted(result["per_table"].items()):
        mark = "OK  " if row["predicted"] == row["true"] else "MISS"
        print(
            f"  {mark} {table:30s} pred={row['predicted']} true={row['true']} "
            f"confidence={row['confidence']:.4f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="refresh data/willibald_eval_features.csv from iceberg.new_predict_data",
    )
    parser.add_argument(
        "--model-uri",
        default="models:/denormalization_model@dev",
        help="MLflow model to score (default: the serving alias)",
    )
    args = parser.parse_args()

    if args.extract:
        extract_snapshot()
        return

    import mlflow  # noqa: PLC0415 - only the scoring path needs MLflow
    from dotenv import dotenv_values  # noqa: PLC0415

    env = dotenv_values(REPO_ROOT / ".env")
    if env.get("MINIO_ROOT_USER") and env.get("MINIO_ROOT_PASSWORD"):
        os.environ.setdefault("AWS_ACCESS_KEY_ID", env["MINIO_ROOT_USER"])
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", env["MINIO_ROOT_PASSWORD"])
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.pyfunc.load_model(args.model_uri)._model_impl.get_raw_model()
    result = evaluate_on_willibald(model)
    if result is None:
        sys.exit(1)
    print_report(result)


if __name__ == "__main__":
    main()
