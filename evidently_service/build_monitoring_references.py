"""
Build the Evidently monitoring baselines for every monitored model.

One run produces, per model, two files:

  evidently_service/references/<track>.csv
      Features + target + prediction + prediction_proba. The drift track reads the
      features and prediction; the classification track reads the target,
      prediction and probability. Baked into the evidently image at build time.

  data/holdouts/<track>.csv
      The labelled source rows NOT used in the reference, left unscored. The
      Prefect backtest replays these through the live model, so the current series
      measures rows the baseline has never seen.

Feature lists are read from the Pydantic request models in webservice/, not
duplicated here. Those models are already the contract the API enforces and the
schema-contract test checks, so deriving from them is what keeps this script from
drifting out of sync with the served schema.

Run once (the model service must be up):

    python evidently_service/build_monitoring_references.py

Re-run after retraining any of these models - a baseline describes how one specific
model version scored, so it is stale the moment that version changes.

    python evidently_service/build_monitoring_references.py --only pk_columns
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parent.parent

# The fk model's cross-table features are computed, not stored, so this script needs the
# same module the training script used - see add_derived_features.
sys.path.insert(0, str(REPO_ROOT))

from src.cross_table_features import add_cross_table_features  # noqa: E402


HERE = Path(__file__).resolve().parent

# One entry per monitored model. The track name is what appears as dataset_name on
# the Prometheus gauges and as the path segment on /iterate/<track>.
# The key models and the normalform model were trained on DIFFERENT column_type
# vocabularies, so the one-hot encoding is not shared. Both dropped the first
# category alphabetically via get_dummies(drop_first=True), which is 'bigint' in
# both cases, but the remaining dummies differ: the key models have 'boolean' and
# no 'char'/'timestamp'; normalform is the other way round. Encoding one model's
# data with the other's vocabulary silently produces the wrong feature set.
KEY_COLUMN_TYPES = ["bigint", "boolean", "date", "decimal", "double", "integer", "varchar"]

# Imported rather than repeated. This file was the fourth place the normalform vocabulary
# lived, and it had drifted: 'char', 'decimal' and 'timestamp' do not occur in the training
# data at all. Phase 2 replaced that data wholesale - see E3 in TASK_3_PLAN.md.
sys.path.insert(0, str(REPO_ROOT / "prefect"))
from nf_features import COLUMN_TYPE_CATEGORIES  # noqa: E402 - path set on the line above


NF_COLUMN_TYPES = list(COLUMN_TYPE_CATEGORIES)

KEY_TRAINING_DATA = "data/summary_output_task_1_2_training.csv"
NF_TRAINING_DATA = "data/nf_training.csv"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "pk_columns": {
        "endpoint": "/predict_pk",
        "module": "data_model_pk",
        "request_model": "PrimaryKey",
        "target": "pk_target",
    },
    "cpk_columns": {
        "endpoint": "/predict_cpk",
        "module": "data_model_cpk",
        "request_model": "CompositePrimaryKey",
        "target": "composite_pk_target",
    },
    "fk_columns": {
        "endpoint": "/predict_fk",
        "module": "data_model_fk",
        "request_model": "ForeignKey",
        "target": "fk_target",
    },
    "cfk_columns": {
        "endpoint": "/predict_cfk",
        "module": "data_model_cfk",
        "request_model": "CompositeForeignKey",
        "target": "composite_fk_target",
    },
    "nf_columns": {
        "endpoint": "/predict_normalform",
        "module": "data_model_denormalization",
        "request_model": "NormalForm",
        "target": "target_normal_form",
        "training_data": NF_TRAINING_DATA,
        "column_types": NF_COLUMN_TYPES,
        # Multiclass (normal forms 0-3). The API returns max(predict_proba), which
        # for >2 classes cannot be expanded back into a full probability vector, so
        # no prediction_proba column is written and log loss is not computable.
        # See create_classification_report in utils.py.
        "multiclass": True,
    },
}

# Applied to any spec that does not override them - the four binary key models.
for _spec in MODEL_SPECS.values():
    _spec.setdefault("training_data", KEY_TRAINING_DATA)
    _spec.setdefault("column_types", KEY_COLUMN_TYPES)
    _spec.setdefault("multiclass", False)

# Pinned rather than inferred: on any subset containing no rows of one type,
# get_dummies would silently emit fewer columns and the payload would no longer
# match the model's expected schema.

# Columns identifying one profiled column in the source warehouse. Hashing these
# rather than the row position keeps the reference/holdout split stable even if the
# training CSV is regenerated in a different order.
IDENTITY_COLUMNS = ["database", "schema", "table_name", "column_name"]


def request_features(spec: dict[str, str]) -> list[str]:
    """Read one model's feature list off its Pydantic request model."""
    sys.path.insert(0, str(REPO_ROOT / "webservice"))
    module = importlib.import_module(spec["module"])
    model = getattr(module, spec["request_model"])
    return list(model.model_fields)


def assign_split(df: pd.DataFrame, reference_pct: int) -> pd.Series:
    """
    Label each row "reference" or "holdout" by hashing its identity columns.

    Deterministic and order-independent, and shared across all four models so a
    given warehouse column is always on the same side of the split everywhere.
    """
    missing = [c for c in IDENTITY_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Training data is missing identity column(s) {missing}."
        raise SystemExit(msg)

    keys = df[IDENTITY_COLUMNS].astype(str).agg("|".join, axis=1)
    buckets = keys.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100)  # noqa: S324
    return pd.Series(
        ["reference" if b < reference_pct else "holdout" for b in buckets],
        index=df.index,
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features the training scripts compute rather than read from the CSV.

    The fk model's seven cross-table features are derived from the identity columns at
    training time (src/cross_table_features.py), so they are absent from the labelled CSV
    and have to be recomputed here or `build_one` would refuse the fk track outright.

    Scoped by `database`, matching the training script. The serving pipeline scopes by
    Trino schema instead and the model is trained to tolerate both widths, but a baseline
    should describe the width the labels were collected in.

    The other tracks ignore the extra columns: each one selects features by name from its
    Pydantic model.
    """
    return add_cross_table_features(df) if "database" in df.columns else df


def encode_column_type(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """One-hot encode column_type exactly as the matching training script does."""
    unexpected = set(df["column_type"].unique()) - set(categories)
    if unexpected:
        msg = (
            f"Unknown column_type value(s) {sorted(unexpected)}. This model was not "
            f"trained on these, so they cannot be encoded."
        )
        raise SystemExit(msg)

    typed = df.copy()
    typed["column_type"] = pd.Categorical(typed["column_type"], categories=categories)
    return pd.get_dummies(typed, columns=["column_type"], drop_first=True)


def to_python(value: object) -> object:
    """numpy scalars are not JSON-serialisable; plain Python ones pass through."""
    return value.item() if hasattr(value, "item") else value


def build_payload(row: pd.Series, features: list[str]) -> dict:
    """Build one prediction request body from an encoded feature row."""
    return {
        key: bool(row[key]) if key.startswith("column_type_") else to_python(row[key])
        for key in features
    }


def score_rows(
    features_df: pd.DataFrame,
    feature_names: list[str],
    endpoint_url: str,
    timeout: float,
    multiclass: bool = False,
) -> pd.DataFrame:
    """POST each row and collect the predicted label plus P(class=1)."""
    session = requests.Session()
    predictions: list[int] = []
    probabilities: list[float] = []
    total = len(features_df)

    for position, (_, row) in enumerate(features_df.iterrows(), start=1):
        response = session.post(
            endpoint_url,
            json=build_payload(row, feature_names),
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        predictions.append(int(body["prediction"]))
        probabilities.append(float(body["probability"]))

        if position % 500 == 0 or position == total:
            logging.info(f"      scored {position}/{total}")

    scored = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities},
        index=features_df.index,
    )

    # The API reports confidence in the *predicted* class, not P(class=1). Log loss
    # needs the latter. For a BINARY argmax classifier the two are related exactly,
    # so this recovers it. Mirrors utils.positive_class_probability - duplicated so
    # this script stays runnable outside the container.
    #
    # For multiclass there is no equivalent: knowing max(proba)=0.7 and the argmax
    # says nothing about how the remaining 0.3 splits across the other classes, so
    # the column is omitted and log loss is simply not offered for those tracks.
    if not multiclass:
        scored["prediction_proba"] = scored.apply(
            lambda r: r["probability"] if int(r["prediction"]) == 1 else 1.0 - r["probability"],
            axis=1,
        )
    return scored


def stratified_sample(df: pd.DataFrame, target: str, limit: int, seed: int) -> pd.DataFrame:
    """
    Sample up to `limit` rows keeping the target's class balance.

    Sampled group by group rather than with groupby().apply(), whose include_groups
    behaviour differs between pandas 2.x and 3.x. This matters most for
    composite_fk_target, which is only ~2% positive: an unstratified sample could
    contain too few positives to score at all.
    """
    if not limit or limit >= len(df):
        return df

    fraction = limit / len(df)
    strata = [
        group.sample(n=max(1, round(len(group) * fraction)), random_state=seed)
        for _, group in df.groupby(target)
    ]
    return pd.concat(strata).reset_index(drop=True)


def build_one(
    track: str,
    spec: dict[str, Any],
    encoded: pd.DataFrame,
    split: pd.Series,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the reference and holdout files for a single model."""
    target = spec["target"]
    multiclass = spec["multiclass"]
    features = request_features(spec)
    flavour = "multiclass" if multiclass else "binary"
    logging.info(f"  {track}: {len(features)} features, target={target} ({flavour})")

    missing = [c for c in features if c not in encoded.columns]
    if missing:
        msg = (
            f"{track}: training data is missing feature column(s) {missing}. The "
            f"Pydantic model and the training data have drifted apart."
        )
        raise SystemExit(msg)

    holdout_source = encoded[split == "holdout"]
    # Keep the raw column_type rather than the dummies: the backtest re-encodes it
    # itself, and storing both would let the two representations disagree.
    holdout_columns = [*IDENTITY_COLUMNS, "column_type", target]
    holdout_columns = [c for c in holdout_columns if c in encoded.columns]
    holdout = pd.concat(
        [holdout_source[holdout_columns], holdout_source[[c for c in features if c in encoded]]],
        axis=1,
    )
    holdout = holdout.loc[:, ~holdout.columns.duplicated()].reset_index(drop=True)

    holdout_path = Path(args.holdout_dir) / f"{track}.csv"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(holdout_path, index=False)

    sampled = stratified_sample(
        encoded[split == "reference"].reset_index(drop=True),
        target,
        args.limit,
        args.seed,
    )

    endpoint_url = f"{args.model_url.rstrip('/')}{spec['endpoint']}"
    scored = score_rows(
        sampled[features],
        features,
        endpoint_url,
        timeout=30.0,
        multiclass=multiclass,
    )

    reference = sampled[features].reset_index(drop=True)
    reference[target] = sampled[target].astype(int).reset_index(drop=True)
    reference["prediction"] = scored["prediction"].reset_index(drop=True)
    if "prediction_proba" in scored.columns:
        reference["prediction_proba"] = scored["prediction_proba"].reset_index(drop=True)

    reference_path = Path(args.reference_dir) / f"{track}.csv"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(reference_path, index=False)

    accuracy = (reference[target] == reference["prediction"]).mean()
    return {
        "track": track,
        "classes": int(reference[target].nunique()),
        "reference_rows": len(reference),
        "holdout_rows": len(holdout),
        "accuracy": accuracy,
        "true_positive_rate": reference[target].mean(),
        "predicted_positive_rate": reference["prediction"].mean(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-url",
        default="http://localhost:8080",
        help="Model service base URL. Use http://model-service:8080 inside the network.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(HERE / "references"),
        help="Directory for the scored baselines (read by the evidently service).",
    )
    parser.add_argument(
        "--holdout-dir",
        default=str(REPO_ROOT / "data" / "holdouts"),
        help="Directory for the unscored holdout rows (read by the Prefect backtest).",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(MODEL_SPECS),
        help="Build just these tracks. Repeatable. Default: all.",
    )
    parser.add_argument(
        "--reference-pct",
        type=int,
        default=50,
        help="Percentage of rows assigned to the reference half; the rest is holdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="Reference rows to score per model. Each is one HTTP round trip.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    tracks = args.only or sorted(MODEL_SPECS)

    # Tracks are grouped by source file: the key models share one labelled CSV, the
    # normalform model has its own. Each is loaded and encoded once.
    results = []
    by_source: dict[str, list[str]] = {}
    for track in tracks:
        by_source.setdefault(MODEL_SPECS[track]["training_data"], []).append(track)

    for source, source_tracks in by_source.items():
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        df = pd.read_csv(source_path)
        logging.info(f"Loaded {len(df)} labelled rows from {source_path}")

        # One encoding per (source, vocabulary). Tracks sharing a source were
        # trained on the same column_type vocabulary, so grouping by source is
        # enough here - the assertion inside encode_column_type catches it if not.
        categories = MODEL_SPECS[source_tracks[0]]["column_types"]
        encoded = encode_column_type(add_derived_features(df), categories)
        split = assign_split(encoded, args.reference_pct)
        logging.info(
            f"  split by identity hash: {(split == 'reference').sum()} reference / "
            f"{(split == 'holdout').sum()} holdout rows",
        )

        for track in source_tracks:
            try:
                results.append(build_one(track, MODEL_SPECS[track], encoded, split, args))
            except requests.RequestException:
                logging.exception(
                    f"{track}: model service call failed. Is it running and is the model "
                    f"registered? Try: docker compose up -d model-service",
                )
                return 1

    logging.info("Baselines built:")
    for r in results:
        logging.info(
            f"  {r['track']:<12} ref={r['reference_rows']:<5} holdout={r['holdout_rows']:<5} "
            f"classes={r['classes']} accuracy={r['accuracy']:.4f}",
        )
    logging.info(
        "Now rebuild the evidently image: docker compose up -d --build evidently_service",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
