"""
Build the baseline dataset for the classification-quality track.

The labelled training data carries ground truth but no predictions, and the model
service returns predictions but no ground truth. F1/precision/recall/log-loss need
both, so this script joins them: it scores the labelled rows through
POST /predict_pk and writes target + predicted label + P(class=1) to a CSV that
the Evidently service loads at startup as its type="reference" series.

Run once (the model service must be up):

    python evidently_service/build_classification_reference.py

Re-run it whenever the pk model is retrained, otherwise the baseline describes a
model that is no longer serving.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests


# Mirrors the PrimaryKey pydantic model in webservice/data_model_pk.py. The
# service rejects payloads with missing or unexpected fields, so this list must
# stay in step with that model.
PK_FEATURES = [
    "number_unique_values",
    "count",
    "is_unique",
    "ordinal_position",
    "unique_ratio",
    "is_first_column",
    "relative_ordinal_position",
    "is_first_unique_column",
    "table_column_count",
    "table_unique_column_count",
    "table_row_count",
    "table_has_unique_column",
    "table_has_no_single_pk_candidate",
    "table_near_unique_column_count",
    "table_id_named_column_count",
    "table_non_null_column_count",
    "table_max_unique_ratio",
    "unique_ratio_rank",
    "null_ratio_rank",
    "is_least_null_in_table",
    "unique_ratio_relative_to_max",
    "name_ends_with_id",
    "name_contains_table_name",
    "name_is_singular_table_id",
    "name_length",
    "column_type_boolean",
    "column_type_date",
    "column_type_decimal",
    "column_type_double",
    "column_type_integer",
    "column_type_varchar",
]

# Training used pd.get_dummies(..., drop_first=True) on column_type, which sorts
# categories alphabetically and drops the first. 'bigint' is therefore the implicit
# baseline and has no column of its own - all six dummies are 0 for those rows.
#
# The full list is pinned here rather than inferred from the data. Inferring it
# breaks on any subset that happens to contain no rows of one type: get_dummies
# would silently emit fewer columns and the payload would no longer match the
# model's expected schema.
COLUMN_TYPE_CATEGORIES = [
    "bigint",
    "boolean",
    "date",
    "decimal",
    "double",
    "integer",
    "varchar",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode column_type exactly as the training script does."""
    unexpected = set(df["column_type"].unique()) - set(COLUMN_TYPE_CATEGORIES)
    if unexpected:
        msg = (
            f"Unknown column_type value(s) {sorted(unexpected)}. The model was not "
            f"trained on these, so they cannot be encoded."
        )
        raise SystemExit(msg)

    # Declaring the dtype as categorical with the full category list makes
    # get_dummies emit every dummy column even for unobserved categories, so the
    # encoding is identical regardless of which rows were sampled.
    typed = df.copy()
    typed["column_type"] = pd.Categorical(
        typed["column_type"],
        categories=COLUMN_TYPE_CATEGORIES,
    )
    encoded = pd.get_dummies(typed, columns=["column_type"], drop_first=True)

    missing = [c for c in PK_FEATURES if c not in encoded.columns]
    if missing:
        msg = (
            f"Training data is missing feature column(s) {missing}. The model "
            f"schema and this script have drifted apart."
        )
        raise SystemExit(msg)

    return encoded


def score_rows(features: pd.DataFrame, model_url: str, timeout: float) -> pd.DataFrame:
    """POST each row to /predict_pk and collect label + positive-class probability."""
    endpoint = f"{model_url.rstrip('/')}/predict_pk"
    predictions: list[int] = []
    probabilities: list[float] = []

    session = requests.Session()
    total = len(features)

    def to_python(value: object) -> object:
        """numpy scalars are not JSON-serialisable; plain Python ones pass through."""
        return value.item() if hasattr(value, "item") else value

    for position, (_, row) in enumerate(features.iterrows(), start=1):
        payload = {
            key: bool(row[key]) if key.startswith("column_type_") else to_python(row[key])
            for key in PK_FEATURES
        }
        response = session.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()

        predictions.append(int(body["prediction"]))
        probabilities.append(float(body["probability"]))

        if position % 250 == 0 or position == total:
            logging.info(f"Scored {position}/{total} rows")

    scored = pd.DataFrame({"prediction": predictions, "probability": probabilities})

    # The API reports confidence in the predicted class; log loss needs P(class=1).
    # Identical conversion to utils.positive_class_probability - kept inline so this
    # script stays runnable outside the container.
    scored["prediction_proba"] = scored.apply(
        lambda r: r["probability"] if int(r["prediction"]) == 1 else 1.0 - r["probability"],
        axis=1,
    )
    return scored


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-data",
        default=str(repo_root / "data" / "summary_output_task_1_2_training.csv"),
        help="Labelled CSV holding both features and the target column.",
    )
    parser.add_argument(
        "--target",
        default="pk_target",
        help="Ground-truth column. Must match classification.target in config.yaml.",
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:8080",
        help="Model service base URL. Use http://model-service:8080 from inside the network.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "classification_reference.csv"),
        help="Where to write the scored baseline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Rows to score. Each row is one HTTP round trip; 0 means all rows.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    df = pd.read_csv(args.training_data)
    logging.info(f"Loaded {len(df)} labelled rows from {args.training_data}")

    if args.target not in df.columns:
        logging.error(f"Target column {args.target!r} not found in the training data.")
        return 1

    # Sample before scoring, and stratify on the target so the baseline keeps the
    # real positive rate. pk_target is only ~12% positive, so an unstratified
    # sample could easily land with too few positives to score meaningfully.
    if args.limit and args.limit < len(df):
        # Sampled group by group rather than with groupby().apply(), whose
        # include_groups behaviour differs between pandas 2.x and 3.x.
        fraction = args.limit / len(df)
        strata = [
            group.sample(n=max(1, round(len(group) * fraction)), random_state=args.seed)
            for _, group in df.groupby(args.target)
        ]
        df = pd.concat(strata).reset_index(drop=True)
        logging.info(f"Stratified sample: {len(df)} rows")

    targets = df[args.target].astype(int).reset_index(drop=True)
    encoded = encode_features(df)

    logging.info(f"Scoring {len(encoded)} rows against {args.model_url} ...")
    try:
        scored = score_rows(encoded[PK_FEATURES], args.model_url, timeout=30.0)
    except requests.RequestException:
        # logging.exception already appends the traceback, so the exception object
        # is not interpolated into the message.
        logging.exception(
            "Model service call failed. Is it running? Try: docker compose up -d model-service",
        )
        return 1

    reference = pd.DataFrame(
        {
            args.target: targets,
            "prediction": scored["prediction"],
            "prediction_proba": scored["prediction_proba"],
        },
    )

    out_path = Path(args.out)
    reference.to_csv(out_path, index=False)

    agreement = (reference[args.target] == reference["prediction"]).mean()
    logging.info(f"Wrote {len(reference)} scored rows to {out_path}")
    logging.info(
        f"Baseline accuracy {agreement:.4f} | "
        f"true positive rate {reference[args.target].mean():.4f} | "
        f"predicted positive rate {reference['prediction'].mean():.4f}",
    )
    logging.info("Restart the evidently service to pick it up as the reference series.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
