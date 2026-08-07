"""
Train several normal-form (denormalization) classifiers, log each as its own
MLflow run, then register the best one (by cross-validated table-level accuracy)
and point the serving alias at it.

This is a MULTICLASS problem (normal-form classes 0..3), so metrics are computed
with average="weighted" and pr_auc uses the full predict_proba matrix.

Candidates (no baseline):
- XGBoost (defaults, balanced sample weights)
- LightGBM (defaults, balanced class weight)
- XGBoost (randomized search, group-aware CV, balanced sample weights)
- LightGBM (randomized search, group-aware CV)

All candidates are logged with the mlflow.sklearn flavor on purpose: the web
service resolves the model via mlflow.pyfunc and then calls
`get_raw_model().predict_proba(...)`. The sklearn flavor keeps the estimator
(and its predict_proba) intact for XGBClassifier and LGBMClassifier; the
xgboost/lightgbm flavors would hand back a Booster with no predict_proba.
"""

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import dotenv_values
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import randint, uniform
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "prefect"))

from nf_features import (  # noqa: E402 - path has to be set before this import
    COLUMN_TYPE_DUMMIES,
    FEATURE_COLUMNS,
    encode_column_type,
)


RSEED = 42

# What links two tables closely enough that they must not straddle the split. See
# `split_groups` - the whole point of Phase 1 lives there.
GROUP_LINK_COLUMNS = ("meta_recipe_id", "meta_pair_id")


def split_groups(df: pd.DataFrame) -> pd.Series:
    """
    One group id per row, such that closely related tables share it.

    `table_name` alone is not enough, and neither is `meta_recipe_id`.

    * Grouping by **table_name** is necessary but trivial: the table-level features are
      identical across a table's rows, so without it near-identical rows sit on both sides.
      It still lets two tables from one recipe - the same shape at a different scale or under
      different naming - split across train and test. Measured cost of that: 0.95 weighted F1
      against 0.77 for the honest split. The gap is memorised shape.
    * Grouping by **meta_recipe_id** fixes that but breaks the matched pairs. A 0NF injection
      and its atomic control differ in exactly one column and carry *different labels* - and
      they carry different recipe ids (`customer_order_columns` against
      `..._control`), so all 24 pairs straddled the split. That is the sharpest leak of the
      three: a near-identical table with the opposite label sitting in training.

    So the groups are the connected components over both links. Union-find rather than a
    composite key because the relation is transitive - one recipe can be tied into a
    component through a pair, and that component has to stay whole.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent.setdefault(item, item) != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for row in df.drop_duplicates("table_name").itertuples():
        table = f"table:{row.table_name}"
        for column in GROUP_LINK_COLUMNS:
            value = getattr(row, column)
            if pd.notna(value):
                union(table, f"{column}:{value}")

    return df["table_name"].map(lambda name: find(f"table:{name}"))


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

MODEL_ARTIFACT_NAME = "denormalization_model"
MODEL_NAME = "denormalization_model"
MODEL_ALIAS = "dev"


def wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int,
) -> ModelVersion:
    """Wait until the registered model version is ready to serve."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        status = model_version.status
        if status == "READY":
            return model_version
        if status == "FAILED_REGISTRATION":
            msg_error_model_version = f"Model version {model_name} v{version} failed registration."
            raise RuntimeError(
                msg_error_model_version,
            )
        time.sleep(1)

    msg_waiting_model = f"Timed out waiting for {model_name} v{version} to become READY."
    raise RuntimeError(
        msg_waiting_model,
    )


def load_data() -> tuple[Path, pd.DataFrame]:
    """
    Load the generated training set (Phase 2 of TASK_3_PLAN.md).

    Replaces `data/nf_test_analyse.csv`, which was hand-labelled, carried no label
    generator, and contained two features copied from the label itself (finding 1.1).
    """
    print("\n------Data Loading------")
    input_path = REPO_ROOT / "data" / "nf_training.csv"
    df = pd.read_csv(input_path)
    print(f"{len(df)} rows, {df['table_name'].nunique()} tables, {len(df.columns)} columns")
    return input_path, df


def one_hot_encode_column_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the `column_type_*` dummies.

    Not `pd.get_dummies(drop_first=True)`: that picks the reference category by alphabetical
    accident and names the dummies after whatever types happen to occur in this file. The
    serving schema has to match both exactly, so the categories and the reference are fixed
    in nf_features and the encoding is shared with the pipeline - see E3 in TASK_3_PLAN.md.
    """
    print("\n------One-Hot Encoding of 'column_type'------")
    df = encode_column_type(df)
    print(f"{list(COLUMN_TYPE_DUMMIES)} (reference category: all-zero)")
    return df


def train_data_train_test_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    np.ndarray,
    pd.Series,
]:
    print("\n------Train Test Split------")

    # Selected, not dropped. The previous version listed 6 identifiers plus 21 redundant
    # features by name; against the generated set that raises KeyError on the first one,
    # because none of those columns exist any more. Selecting from the shared contract also
    # means a new feature reaches the model by appearing in FEATURE_COLUMNS and nowhere else.
    feature_names = [*FEATURE_COLUMNS, *COLUMN_TYPE_DUMMIES]
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        message = f"training set is missing features the contract requires: {missing}"
        raise KeyError(message)

    X = df[feature_names]
    y = df["target_normal_form"].astype(int)

    print(f"Feature count: {X.shape[1]}")

    groups = split_groups(df)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # All three counts, so the split's difficulty is visible instead of implied - the same
    # model scores very differently under each. See split_groups.
    print(f"Split groups (recipe + matched pair): {groups.nunique()}")
    print(f"  by meta_recipe_id alone:            {df['meta_recipe_id'].nunique()}")
    print(f"  by table_name alone:                {df['table_name'].nunique()}")
    train_tables = groups.iloc[train_idx].nunique()
    test_tables = groups.iloc[test_idx].nunique()
    print(f"Groups in train: {train_tables} | in test: {test_tables}")
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    for nf_class in sorted(y_train.unique()):
        train_pct = (y_train == nf_class).mean() * 100
        test_pct = (y_test == nf_class).mean() * 100
        print(
            f"Train {nf_class} NF rate: {round(train_pct, 2)}% | "
            f"Test {nf_class} NF rate: {round(test_pct, 2)}%",
        )

    return X, y, X_train, X_test, y_train, y_test, train_idx, groups


def print_x_y_shape(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    print("\n------Print X, Y Shape------")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)


def print_denormlization_target_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    print("\n------Target Distribution------")
    print(y_train.value_counts())
    print(y_test.value_counts())


def predict_xg_boost(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict XGBoost Model------")
    # Multiclass imbalance is handled with per-sample weights (scale_pos_weight is
    # binary-only).
    sample_weights_train = compute_sample_weight("balanced", y=y_train)

    model_xgb = xgb.XGBClassifier(random_state=RSEED)
    model_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)

    y_pred_xgb_train = model_xgb.predict(X_train)
    y_pred_xgb_test = model_xgb.predict(X_test)
    y_pred_proba_xgb_train = model_xgb.predict_proba(X_train)
    y_pred_proba_xgb_test = model_xgb.predict_proba(X_test)

    return (
        model_xgb,
        y_pred_xgb_train,
        y_pred_xgb_test,
        y_pred_proba_xgb_train,
        y_pred_proba_xgb_test,
    )


def predict_light_gbm(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict LightGBM Model------")
    model_lgbm = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=RSEED,
        verbose=-1,
    )
    model_lgbm.fit(X_train, y_train)

    y_pred_lgbm_train = model_lgbm.predict(X_train)
    y_pred_lgbm_test = model_lgbm.predict(X_test)
    y_pred_proba_lgbm_train = model_lgbm.predict_proba(X_train)
    y_pred_proba_lgbm_test = model_lgbm.predict_proba(X_test)

    return (
        model_lgbm,
        y_pred_lgbm_train,
        y_pred_lgbm_test,
        y_pred_proba_lgbm_train,
        y_pred_proba_lgbm_test,
    )


def predict_xg_boost_randomized_search(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_idx: np.ndarray,
    groups: pd.Series,
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict XGBoost Model with hyperparameter search------")

    param_distributions_xgb = {
        "max_depth": randint(3, 8),
        "n_estimators": randint(50, 600),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(5, 50),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 4.5),
        "gamma": uniform(0.0, 0.5),
    }

    model_xgb_grid = xgb.XGBClassifier(random_state=RSEED)

    randomized_search_xgb = RandomizedSearchCV(
        estimator=model_xgb_grid,
        param_distributions=param_distributions_xgb,
        n_iter=100,
        cv=StratifiedGroupKFold(n_splits=5),
        verbose=1,
        scoring="f1_weighted",
        random_state=RSEED,
        n_jobs=-1,
    )

    sample_weights_train = compute_sample_weight("balanced", y=y_train)
    randomized_search_xgb.fit(
        X_train,
        y_train,
        groups=groups.iloc[train_idx],
        sample_weight=sample_weights_train,
    )

    print("------Best Score:------")
    print("Best score: ", randomized_search_xgb.best_score_)
    print("------Best Hyperparameters:------")
    print(randomized_search_xgb.best_params_)

    best_estimator = randomized_search_xgb.best_estimator_

    y_pred_train = randomized_search_xgb.predict(X_train)
    y_pred_test = randomized_search_xgb.predict(X_test)
    y_pred_proba_train = randomized_search_xgb.predict_proba(X_train)
    y_pred_proba_test = randomized_search_xgb.predict_proba(X_test)

    return (
        best_estimator,
        y_pred_train,
        y_pred_test,
        y_pred_proba_train,
        y_pred_proba_test,
    )


def predict_light_gbm_randomized_search(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_idx: np.ndarray,
    groups: pd.Series,
) -> tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict LightGBM Model with hyperparameter search------")

    param_distributions_lgbm = {
        "num_leaves": randint(4, 64),
        "n_estimators": randint(100, 600),
        "learning_rate": uniform(0.01, 0.19),
        "min_child_samples": randint(10, 80),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 5.0),
        "subsample": uniform(0.6, 0.4),
    }

    model_lgbm_grid = lgb.LGBMClassifier(
        class_weight="balanced",
        random_state=RSEED,
        verbose=-1,
        n_jobs=1,
    )

    randomized_search_lgbm = RandomizedSearchCV(
        estimator=model_lgbm_grid,
        param_distributions=param_distributions_lgbm,
        n_iter=50,
        cv=StratifiedGroupKFold(n_splits=3),
        verbose=1,
        scoring="f1_weighted",
        random_state=RSEED,
        n_jobs=-1,
    )

    randomized_search_lgbm.fit(X_train, y_train, groups=groups.iloc[train_idx])

    print("------Best Score:------")
    print("Best score: ", randomized_search_lgbm.best_score_)
    print("------Best Hyperparameters:------")
    print(randomized_search_lgbm.best_params_)

    best_estimator = randomized_search_lgbm.best_estimator_

    y_pred_train = randomized_search_lgbm.predict(X_train)
    y_pred_test = randomized_search_lgbm.predict(X_test)
    y_pred_proba_train = randomized_search_lgbm.predict_proba(X_train)
    y_pred_proba_test = randomized_search_lgbm.predict_proba(X_test)

    return (
        best_estimator,
        y_pred_train,
        y_pred_test,
        y_pred_proba_train,
        y_pred_proba_test,
    )


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Weighted multiclass precision, recall, F1, accuracy, and PR-AUC.

    y_pred_proba is the full (n_samples, n_classes) probability matrix; pr_auc uses
    average="weighted" over the one-vs-rest curves.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba, average="weighted")),
    }


def _make_candidate(
    name: str,
    model_type: str,
    model: ClassifierMixin,
    y_train: pd.Series,
    y_test: pd.Series,
    preds: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    """Package a trained model with its train/test metrics and confusion matrices."""
    y_pred_train, y_pred_test, y_pred_proba_train, y_pred_proba_test = preds
    return {
        "name": name,
        "model_type": model_type,
        "model": model,
        "train_metrics": compute_metrics(y_train, y_pred_train, y_pred_proba_train),
        "test_metrics": compute_metrics(y_test, y_pred_test, y_pred_proba_test),
        "train_confusion_matrix": confusion_matrix(y_train, y_pred_train),
        "test_confusion_matrix": confusion_matrix(y_test, y_pred_test),
    }


def train_all_candidates(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_idx: np.ndarray,
    groups: pd.Series,
) -> list[dict]:
    """Train every candidate model and package each with its train/test metrics."""
    print("\n------Training candidate models------")

    candidates: list[dict] = []

    model_xgb, *xgb_preds = predict_xg_boost(RSEED, X_train, y_train, X_test)
    candidates.append(
        _make_candidate("xgboost", "xgboost_classifier", model_xgb, y_train, y_test, xgb_preds),
    )

    model_lgbm, *lgbm_preds = predict_light_gbm(RSEED, X_train, y_train, X_test)
    candidates.append(
        _make_candidate(
            "lightgbm",
            "lightgbm_classifier",
            model_lgbm,
            y_train,
            y_test,
            lgbm_preds,
        ),
    )

    best_xgb, *xgb_rs_preds = predict_xg_boost_randomized_search(
        RSEED,
        X_train,
        y_train,
        X_test,
        train_idx,
        groups,
    )
    candidates.append(
        _make_candidate(
            "xgboost_randomized_search",
            "xgboost_classifier",
            best_xgb,
            y_train,
            y_test,
            xgb_rs_preds,
        ),
    )

    best_lgbm, *lgbm_rs_preds = predict_light_gbm_randomized_search(
        RSEED,
        X_train,
        y_train,
        X_test,
        train_idx,
        groups,
    )
    candidates.append(
        _make_candidate(
            "lightgbm_randomized_search",
            "lightgbm_classifier",
            best_lgbm,
            y_train,
            y_test,
            lgbm_rs_preds,
        ),
    )

    return candidates


N_SPLITS = 5

# Where the cross-validated baseline is written after every training run.
#
# Committed, and guarded by test/test_models/test_normalform_baseline.py. CI has neither
# MLflow nor Trino, so it cannot retrain to check for a regression - but it can check that
# the recorded baseline still clears the floors the plan claims. Lowering a floor then
# becomes a visible edit in a diff instead of a number quietly getting worse.
BASELINE_PATH = REPO_ROOT / "data" / "nf_baseline.json"


def evaluate_across_folds(
    candidate: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    table_names: pd.Series,
) -> dict[str, float]:
    """
    Refit this candidate's configuration on every fold and report the spread.

    Run for every candidate, not just the single-split winner: with 327 groups and 65 of
    them in a holdout, one train/test split is one draw, and picking the "best" candidate
    by that draw risks picking the one that got lucky rather than the one that generalises.
    Five folds give a mean and a standard deviation per candidate, which is what the
    selection in `main` actually compares.

    Both units are reported, and the table-level one is the honest headline: the pipeline
    stores one normal form per table, and a 40-column table otherwise counts forty times as
    much as a five-column one in the row-level metric.
    """
    print(f"\n------Cross-validating '{candidate['name']}' across {N_SPLITS} folds------")
    estimator = candidate["model"]
    column_scores: list[float] = []
    table_scores: list[float] = []

    # Balance the way the winning candidate was originally trained, not the way that is
    # convenient here. LightGBM carries `class_weight="balanced"` in its constructor and is
    # fitted without sample weights; passing them anyway multiplies the two, so the
    # cross-validated number would describe a configuration that was never registered.
    # XGBoost has no class_weight and takes per-sample weights instead.
    balances_itself = getattr(estimator, "class_weight", None) is not None

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RSEED)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        fold_model = clone(estimator)
        fit_kwargs = (
            {}
            if balances_itself
            else {"sample_weight": compute_sample_weight("balanced", y=y.iloc[train_idx])}
        )
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        predicted = fold_model.predict(X.iloc[test_idx])

        column_f1 = float(f1_score(y.iloc[test_idx], predicted, average="weighted"))
        votes = pd.DataFrame(
            {
                "table": table_names.iloc[test_idx].to_numpy(),
                "true": y.iloc[test_idx].to_numpy(),
                "pred": predicted,
            },
        )
        per_table = votes.groupby("table").agg(
            true=("true", "first"),
            voted=("pred", lambda column: column.value_counts().idxmax()),
        )
        table_accuracy = float((per_table["true"] == per_table["voted"]).mean())

        column_scores.append(column_f1)
        table_scores.append(table_accuracy)
        print(
            f"  fold {fold}: column-level F1 {column_f1:.4f} | "
            f"table-level accuracy {table_accuracy:.4f} ({len(per_table)} tables)",
        )

    summary = {
        "cv_column_f1_mean": float(np.mean(column_scores)),
        "cv_column_f1_std": float(np.std(column_scores)),
        "cv_table_accuracy_mean": float(np.mean(table_scores)),
        "cv_table_accuracy_std": float(np.std(table_scores)),
    }
    print(
        f"\n  column-level F1     {summary['cv_column_f1_mean']:.4f} "
        f"+/- {summary['cv_column_f1_std']:.4f}",
    )
    print(
        f"  table-level accuracy {summary['cv_table_accuracy_mean']:.4f} "
        f"+/- {summary['cv_table_accuracy_std']:.4f}   <- the honest headline",
    )
    return summary


def write_baseline(
    candidate: dict,
    cv_summary: dict[str, float],
    df: pd.DataFrame,
    groups: pd.Series,
    n_features: int,
) -> None:
    """Record the cross-validated baseline so CI can guard it without retraining."""
    baseline = {
        "candidate": candidate["name"],
        "n_rows": len(df),
        "n_tables": int(df["table_name"].nunique()),
        "n_split_groups": int(groups.nunique()),
        "n_features": int(n_features),
        "n_splits": N_SPLITS,
        "split_group_links": list(GROUP_LINK_COLUMNS),
        **{name: round(value, 4) for name, value in cv_summary.items()},
        "single_split_column_f1": round(candidate["test_metrics"]["f1_score"], 4),
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"\nBaseline written to {BASELINE_PATH.relative_to(REPO_ROOT)}")


def print_candidate_summary(candidates: list[dict]) -> None:
    """Print a side-by-side metrics table and confusion matrices for all candidates."""
    print("\n------Evaluation Table Test Set------\n")
    summary = pd.DataFrame(
        {c["name"]: c["test_metrics"] for c in candidates},
    ).T
    print(summary)

    print("\n------Confusion Matrices (Test Set)------\n")
    for c in candidates:
        print(f"{c['name']}:")
        print(c["test_confusion_matrix"])


def print_cv_comparison(candidates: list[dict]) -> None:
    """
    Side-by-side cross-validated metrics for every candidate - this is what selection
    actually reads, as opposed to the single-split table above which is diagnostic only.
    """
    print("\n------Cross-Validated Comparison (all candidates, all folds)------\n")
    summary = pd.DataFrame(
        {c["name"]: c["cv_summary"] for c in candidates},
    ).T
    print(summary.sort_values("cv_table_accuracy_mean", ascending=False))


def setup_experiment(client: MlflowClient, experiment_name: str) -> str:
    """Create the experiment if needed and return its id."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_candidate_run(
    candidate: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    input_path: Path,
    cv_summary: dict[str, float] | None = None,
) -> str:
    """Log one candidate as its own MLflow run and return the run id.

    The model artifact is logged with the sklearn flavor (works for XGBClassifier
    and LGBMClassifier) so the serving side can call predict_proba on the raw model.
    """
    input_example = X_train.head(5).astype(float)
    model = candidate["model"]
    signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run(run_name=candidate["name"]) as run:
        mlflow.log_params(model.get_params())

        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("holdout_rows", len(X_test))
        mlflow.log_param("input_path", str(input_path))
        # Provenance for the metric: the same model scores 0.95 or 0.77 depending on this.
        mlflow.log_param("split_group_links", ",".join(GROUP_LINK_COLUMNS))
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("alias", MODEL_ALIAS)
        mlflow.log_param("candidate", candidate["name"])

        for metric_name, value in candidate["train_metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in candidate["test_metrics"].items():
            mlflow.log_metric(f"test_{metric_name}", value)
        # Every candidate carries these now - selection reads them, so every run should show
        # the number it was picked (or passed over) on: a mean over five folds, plus the
        # table-level unit the pipeline actually delivers.
        for metric_name, value in (cv_summary or {}).items():
            mlflow.log_metric(metric_name, value)

        mlflow.set_tags(
            {
                "model_type": candidate["model_type"],
                "candidate": candidate["name"],
                "developer": "test",
                "dataset": "generated-denormalization-statistics",
                "target_column": y.name,
                "n_features": len(X.columns),
                "n_samples": len(X),
            },
        )

        mlflow.log_dict(
            {"features": list(X.columns)},
            "features.json",
        )

        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            serialization_format="pickle",
            signature=signature,
            input_example=input_example,
        )

    test_f1 = candidate["test_metrics"]["f1_score"]
    name = candidate["name"]
    print(f"Logged run {run.info.run_id} for '{name}' (test weighted-F1={test_f1:.4f})")
    return run.info.run_id


def select_best_candidate(candidates: list[dict]) -> dict:
    """
    Pick the candidate with the highest cross-validated table-level accuracy.

    Not the single-split test F1: that number is one draw out of 327 groups, and with 65
    of them in a holdout it is exactly the kind of figure `evaluate_across_folds` exists to
    replace. Every candidate now carries a `cv_summary` (mean +/- std over 5 group-aware
    folds), so selection reads the same honest headline that ends up in the baseline and in
    the CI gate - not a noisier proxy for it. Table-level, not column-level, because that is
    the unit the pipeline actually delivers (one normal form per table via majority vote).
    """
    return max(candidates, key=lambda c: c["cv_summary"]["cv_table_accuracy_mean"])


def register_best_candidate(
    best: dict,
    run_ids: dict[str, str],
    model_name: str,
    alias: str,
) -> ModelVersion:
    """Register the given (already-selected) candidate and point the alias at it."""
    print("\n------MLflow Model Registration------")

    best_run_id = run_ids[best["name"]]
    best_table_acc = best["cv_summary"]["cv_table_accuracy_mean"]
    best_table_std = best["cv_summary"]["cv_table_accuracy_std"]
    model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_NAME}"

    print(
        f"Best candidate: '{best['name']}' with CV table-accuracy="
        f"{best_table_acc:.4f} +/- {best_table_std:.4f}",
    )
    print(f"Registering {model_uri} as {model_name}")

    client = MlflowClient()

    registration = mlflow.register_model(model_uri=model_uri, name=model_name)
    model_version = wait_for_model_version(
        client=client,
        model_name=model_name,
        version=registration.version,
        timeout_seconds=60,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version.version,
    )

    print(
        f"Registered {model_name} version {model_version.version} and assigned alias '{alias}'.",
    )
    print(
        "The FastAPI service can now resolve models:/"
        f"{model_name}@{alias} from your local MLflow server.",
    )

    return model_version


def main() -> None:
    input_path, df = load_data()
    df = one_hot_encode_column_type(df)

    # Train Test Split
    (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        train_idx,
        groups,
    ) = train_data_train_test_split(df)

    print_x_y_shape(X_train, X_test, y_train, y_test)
    print_denormlization_target_distribution(y_train, y_test)

    # Train all candidate models and score them.
    candidates = train_all_candidates(
        RSEED,
        X_train,
        y_train,
        X_test,
        y_test,
        train_idx,
        groups,
    )
    print_candidate_summary(candidates)

    # Every candidate is cross-validated, not just the single-split winner - selection has
    # to read the same honest, low-variance number that ends up in the registry and the CI
    # gate, not the noisier single-split F1 that produced it before.
    for candidate in candidates:
        candidate["cv_summary"] = evaluate_across_folds(candidate, X, y, groups, df["table_name"])
    print_cv_comparison(candidates)

    best = select_best_candidate(candidates)
    write_baseline(best, best["cv_summary"], df, groups, n_features=X.shape[1])

    # Log every candidate as its own run under one experiment.
    client = MlflowClient()
    setup_experiment(client, "denormalization_model_training")

    run_ids: dict[str, str] = {}
    for candidate in candidates:
        run_ids[candidate["name"]] = log_candidate_run(
            candidate=candidate,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            input_path=input_path,
            cv_summary=candidate["cv_summary"],
        )

    # Register the best candidate (by CV table-accuracy) and move the serving alias.
    register_best_candidate(
        best=best,
        run_ids=run_ids,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    main()
