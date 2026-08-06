"""
Train several composite-primary-key classifiers, score them with grouped
cross-validation, then register the best one (by CV mean F1) and point the serving
alias at it.

Candidates (no baseline):
- XGBoost (defaults, scale_pos_weight)
- Random forest (defaults, balanced class weight)
- XGBoost (randomized search + tuned decision threshold)
- Random forest (randomized search + tuned decision threshold)

LightGBM is deliberately absent here (the fk/cfk scripts do include it). Adding it is
two entries in CANDIDATE_SPECS plus one in PARAM_DISTRIBUTIONS; it was left out of the
evaluation rewrite so that a change in the registered model can only come from the
fixed measurement, not from a new contender.

Why grouped cross-validation instead of one holdout
---------------------------------------------------
Until 2026-08-05 this script scored every candidate on a single fold, and it picked
that fold by accident:

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set()
    X_train = X.iloc[train_idx]   # whatever the loop left behind - the LAST fold

The loop existed only to assert that no database straddles a split, but the training
data was then taken from the leftover loop variable. Across the five folds a plain
LightGBM on this target swings between F1 0.60 and 0.78, and the leftover fold sits at
the weak end of that range, so the reported quality was both noisy and pessimistic.

Two grouping leaks went with it. The randomized search ran on `cv=3` - a plain,
ungrouped KFold, with no `groups` argument - and the threshold tuning on `cv=5`, also
ungrouped. Both put columns of the same database on either side of a selection split,
so hyperparameters and decision threshold were partly chosen on memorised rows.

Every split in this script now comes from one materialised StratifiedGroupKFold over
`database`, which is also what makes the old assertion unnecessary: disjointness holds
by construction. Candidates are scored on pooled out-of-fold predictions over all five
folds, giving an honest point estimate (`cv_f1_mean`) next to its spread
(`cv_f1_std`); selection uses the mean. The confusion matrix logged to MLflow is built
from those pooled predictions, so it covers every labelled row exactly once instead of
a fifth of them.

Hyperparameter search runs ONCE per model family, on the first fold's training rows,
and the parameters it finds are then scored across all five folds. That is not a full
nested CV - those parameters have seen four fifths of the data - but a nested search
costs five times the runtime for a second-order correction. The decision threshold, a
far stronger leak, IS tuned inside each fold; see fit_candidate.

All candidates are logged with the mlflow.sklearn flavor on purpose: the web service
resolves the model via mlflow.pyfunc and then calls
`get_raw_model().predict_proba(...)`. The sklearn flavor keeps the estimator (and its
predict_proba) intact for RandomForest, XGBClassifier and the
TunedThresholdClassifierCV wrapper; the xgboost flavor would hand back a Booster with
no predict_proba.

For the tuned candidates the TunedThresholdClassifierCV itself is registered (not the
raw best_estimator_), so the served model applies the same decision threshold that
produced the reported metrics.
"""

import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import dotenv_values
from matplotlib.figure import Figure
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import randint, uniform
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    TunedThresholdClassifierCV,
)


RSEED = 42

# One number for every fold count in this script: the outer scoring CV, the inner CV
# of the hyperparameter search, and the inner CV of the threshold tuning.
N_SPLITS = 5

# Sampled hyperparameter settings per family. The search is the dominant cost of this
# script (N_ITER * N_SPLITS fits per family); 30 is enough to beat the defaults, and
# the budget freed up pays for scoring across all five folds instead of one.
SEARCH_N_ITER = 30

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

MODEL_ARTIFACT_NAME = "composite_pk_model"
MODEL_NAME = "composite_pk_model"
MODEL_ALIAS = "dev"
EXPERIMENT_NAME = "composite_pk_model_training"

TARGET_COLUMN = "composite_pk_target"

# Axis labels for the logged confusion matrix.
NEGATIVE_LABEL = "not cpk"
POSITIVE_LABEL = "cpk"

# Only the identity columns are dropped: unlike the fk model, this target keeps the
# full feature set.
COLUMNS_TO_DROP = [
    "database",
    "schema",
    "table_name",
    "column_name",
    "min_value",
    "max_value",
]

# Every target column has to leave the feature matrix, whichever one is being modelled.
TARGET_COLUMNS = [
    "pk_target",
    "composite_pk_target",
    "fk_target",
    "composite_fk_target",
]

# Name-based features whose rate is reported per error bucket in the error analysis.
# These are the ones a human can sanity-check against a column name.
NAME_FEATURES_FOR_ERROR_ANALYSIS = [
    "name_ends_with_id",
    "name_contains_key",
    "name_contains_table_name",
    "name_is_singular_table_id",
]

# How many example column names to keep per error bucket in the logged analysis.
ERROR_ANALYSIS_SAMPLE_SIZE = 30

PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "xgboost": {
        "max_depth": randint(3, 8),
        "n_estimators": randint(50, 500),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 5),
    },
    "random_forest": {
        "n_estimators": randint(100, 600),
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    },
}

# `searched` decides whether the family's hyperparameter search result is applied;
# `tuned` decides whether the decision threshold is tuned. They move together today
# but are separate flags so a searched-but-untuned variant costs one line to add.
CANDIDATE_SPECS: list[dict] = [
    {
        "name": "xgboost",
        "model_type": "xgboost_classifier",
        "family": "xgboost",
        "searched": False,
        "tuned": False,
    },
    {
        "name": "random_forest",
        "model_type": "random_forest_classifier",
        "family": "random_forest",
        "searched": False,
        "tuned": False,
    },
    {
        "name": "xgboost_randomized_search",
        "model_type": "xgboost_classifier_tuned_threshold",
        "family": "xgboost",
        "searched": True,
        "tuned": True,
    },
    {
        "name": "random_forest_randomized_search",
        "model_type": "random_forest_classifier_tuned_threshold",
        "family": "random_forest",
        "searched": True,
        "tuned": True,
    },
]


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
    print("\n------Data Loading------")
    current_pwd = Path.cwd().resolve()
    train_data_path = "data/summary_output_task_1_2_training.csv"
    input_path = current_pwd / train_data_path
    df = pd.read_csv(input_path)
    print(df.head())
    return input_path, df


def one_hot_encode_column_type(df: pd.DataFrame) -> pd.DataFrame:
    print("\n------Column Preview and One-Hot Encoding of 'column_type'------")
    df = pd.get_dummies(df, columns=["column_type"], drop_first=True)
    print(df.columns)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the frame into features, target and the grouping key.

    `groups` is the database name. Every split in this script is grouped by it, so no
    database contributes rows to both sides of any train/test boundary.
    """
    print("\n------Feature Matrix------")

    groups = df["database"]
    X = df.drop(columns=[*TARGET_COLUMNS, *COLUMNS_TO_DROP])
    y = df[TARGET_COLUMN]

    print(f"Feature count: {X.shape[1]}")
    print(X.info())
    print(f"Rows: {len(X)} | databases: {groups.nunique()}")
    print(f"Positive rate ({TARGET_COLUMN}): {round(y.mean() * 100, 1)}%")

    return X, y, groups


def make_grouped_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise the outer CV folds once so every candidate is scored on the same ones.

    Comparing candidates only means something if they saw identical folds, and a
    generator would be consumed by the first candidate.
    """
    print("\n------Grouped Cross-Validation Folds------")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RSEED)
    splits = list(sgkf.split(X, y, groups=groups))

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(
            f"fold {fold}: train={len(train_idx)} rows / "
            f"{groups.iloc[train_idx].nunique()} databases | "
            f"test={len(test_idx)} rows / {groups.iloc[test_idx].nunique()} databases | "
            f"test positive rate={round(y.iloc[test_idx].mean() * 100, 1)}%",
        )

    return splits


def make_base_estimator(family: str, y_train: pd.Series) -> ClassifierMixin:
    """Build an untrained estimator with this family's class-imbalance handling.

    XGBoost has no `class_weight`, so the positive class is upweighted through
    scale_pos_weight instead - recomputed from the rows it is about to be fitted on,
    which is why this takes y_train rather than being a module-level constant.
    """
    if family == "xgboost":
        pos = int(y_train.sum())
        neg = int((y_train == 0).sum())
        return xgb.XGBClassifier(
            scale_pos_weight=neg / max(pos, 1),
            random_state=RSEED,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            random_state=RSEED,
            class_weight="balanced",
        )

    msg_unknown_family = f"Unknown model family: {family}"
    raise ValueError(msg_unknown_family)


def search_hyperparameters(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    search_idx: np.ndarray,
) -> dict:
    """Randomized search for one family, scored by grouped CV inside `search_idx`.

    Run on the first outer fold's training rows only, so the search never sees that
    fold's test databases. It does see the test databases of the other four folds -
    the compromise the module docstring describes.

    Scoring is average_precision rather than f1 because it is threshold-free: the
    decision threshold is a separate, later decision (see fit_candidate), and ranking
    hyperparameters by a metric that depends on the default 0.5 cut would conflate
    the two.
    """
    print(f"\n------Hyperparameter search: {family}------")
    X_train = X.iloc[search_idx]
    y_train = y.iloc[search_idx]
    groups_train = groups.iloc[search_idx]

    search = RandomizedSearchCV(
        estimator=make_base_estimator(family, y_train),
        param_distributions=PARAM_DISTRIBUTIONS[family],
        n_iter=SEARCH_N_ITER,
        scoring="average_precision",
        cv=StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RSEED),
        n_jobs=-1,
        random_state=RSEED,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups_train)

    print(f"Best average_precision: {search.best_score_:.4f}")
    print(f"Best hyperparameters: {search.best_params_}")

    return dict(search.best_params_)


def fit_candidate(
    spec: dict,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> ClassifierMixin:
    """Fit one candidate on the rows it is allowed to see.

    For tuned candidates the decision threshold is chosen by an inner grouped CV over
    `X_train` only. Before 2026-08-05 this was `cv=5`, a plain KFold, which put the
    same database on both sides of the threshold-selection split and produced an
    optimistic threshold; the F1 it was tuned for was partly memorised.
    """
    estimator = make_base_estimator(spec["family"], y_train)
    if params:
        estimator.set_params(**params)

    if not spec["tuned"]:
        estimator.fit(X_train, y_train)
        return estimator

    inner_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RSEED)
    tuned = TunedThresholdClassifierCV(
        estimator,
        scoring="f1",
        cv=list(inner_cv.split(X_train, y_train, groups=groups_train)),
    )
    tuned.fit(X_train, y_train)
    return tuned


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Precision, recall, F1, accuracy, and PR-AUC for one set of predictions."""
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
    }


def cross_validate_candidate(
    spec: dict,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Score one candidate out-of-fold across every grouped fold.

    StratifiedGroupKFold uses each row as test exactly once, so the pooled
    predictions cover the whole labelled set and the confusion matrix built from them
    accounts for every row.

    Returns per-fold F1 (the spread is the point of this whole exercise) alongside the
    pooled metrics.
    """
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_f1: list[float] = []
    fold_thresholds: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        model = fit_candidate(
            spec,
            params,
            X.iloc[train_idx],
            y.iloc[train_idx],
            groups.iloc[train_idx],
        )
        y_pred = model.predict(X.iloc[test_idx])
        y_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        oof_pred[test_idx] = y_pred
        oof_proba[test_idx] = y_proba

        fold_score = float(f1_score(y.iloc[test_idx], y_pred))
        fold_f1.append(fold_score)
        if isinstance(model, TunedThresholdClassifierCV):
            fold_thresholds.append(float(model.best_threshold_))

        print(f"  fold {fold}: F1={fold_score:.4f}")

    return {
        "fold_f1": fold_f1,
        "cv_f1_mean": float(np.mean(fold_f1)),
        "cv_f1_std": float(np.std(fold_f1)),
        "cv_f1_min": float(np.min(fold_f1)),
        "cv_f1_max": float(np.max(fold_f1)),
        "fold_thresholds": fold_thresholds,
        "oof_metrics": compute_metrics(y, oof_pred, oof_proba),
        "oof_confusion_matrix": confusion_matrix(y, oof_pred),
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
    }


def build_candidate(
    spec: dict,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Cross-validate a candidate, then refit it on every row for serving.

    The registered model is the full-data refit, because throwing away a fifth of the
    training rows to keep a holdout buys nothing once the honest estimate comes from
    the out-of-fold predictions. Its reported quality is the CV estimate, not the
    in-sample metrics - those are logged only as an overfitting tell.
    """
    print(f"\n------Candidate: {spec['name']}------")

    cv_result = cross_validate_candidate(spec, params, X, y, groups, splits)
    print(
        f"  cv_f1_mean={cv_result['cv_f1_mean']:.4f} "
        f"(std={cv_result['cv_f1_std']:.4f}, "
        f"min={cv_result['cv_f1_min']:.4f}, max={cv_result['cv_f1_max']:.4f})",
    )

    final_model = fit_candidate(spec, params, X, y, groups)
    if isinstance(final_model, TunedThresholdClassifierCV):
        print(f"  final tuned threshold: {final_model.best_threshold_:.3f}")

    in_sample_pred = final_model.predict(X)
    in_sample_proba = final_model.predict_proba(X)[:, 1]

    return {
        "name": spec["name"],
        "model_type": spec["model_type"],
        "model": final_model,
        "params": params,
        "cv": cv_result,
        "cv_metrics": cv_result["oof_metrics"],
        "train_metrics": compute_metrics(y, in_sample_pred, in_sample_proba),
    }


def train_all_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict]:
    """Search hyperparameters once per family, then cross-validate every candidate."""
    print("\n------Training candidate models------")

    search_idx = splits[0][0]
    searched_families = {spec["family"] for spec in CANDIDATE_SPECS if spec["searched"]}
    best_params = {
        family: search_hyperparameters(family, X, y, groups, search_idx)
        for family in sorted(searched_families)
    }

    return [
        build_candidate(
            spec,
            best_params[spec["family"]] if spec["searched"] else {},
            X,
            y,
            groups,
            splits,
        )
        for spec in CANDIDATE_SPECS
    ]


def print_candidate_summary(candidates: list[dict]) -> None:
    """Print the CV comparison table plus pooled out-of-fold confusion matrices."""
    print("\n------Cross-Validation Comparison------\n")
    summary = pd.DataFrame(
        [
            {
                "candidate": c["name"],
                "cv_f1_mean": round(c["cv"]["cv_f1_mean"], 4),
                "cv_f1_std": round(c["cv"]["cv_f1_std"], 4),
                "cv_f1_min": round(c["cv"]["cv_f1_min"], 4),
                "cv_f1_max": round(c["cv"]["cv_f1_max"], 4),
                "oof_precision": round(c["cv_metrics"]["precision"], 4),
                "oof_recall": round(c["cv_metrics"]["recall"], 4),
                "oof_pr_auc": round(c["cv_metrics"]["pr_auc"], 4),
                "train_f1": round(c["train_metrics"]["f1_score"], 4),
            }
            for c in candidates
        ],
    ).set_index("candidate")
    print(summary.sort_values("cv_f1_mean", ascending=False))

    print("\n------Pooled Out-Of-Fold Confusion Matrices [[TN FP], [FN TP]]------\n")
    for c in candidates:
        print(f"{c['name']} (per-fold F1: {[round(f, 3) for f in c['cv']['fold_f1']]}):")
        print(c["cv"]["oof_confusion_matrix"])


def analyse_errors(df: pd.DataFrame, y: pd.Series, oof_pred: np.ndarray) -> dict:
    """Describe what the best candidate gets wrong, from the pooled OOF predictions.

    This is the artifact that answers "where exactly does it hurt": which column names
    are missed, which are wrongly flagged, and how the name-based features behave in
    each bucket. A confusion matrix says how many; this says which.

    `sibling_target_rate` counts false positives that are really some other kind of
    key - a single PK, an FK, or part of a composite FK. A high rate there means the
    model cannot tell the key roles apart, which is a feature problem rather than a
    threshold problem.
    """
    frame = df.assign(_pred=oof_pred, _true=y.to_numpy())
    buckets = {
        "false_negative": frame[(frame["_true"] == 1) & (frame["_pred"] == 0)],
        "false_positive": frame[(frame["_true"] == 0) & (frame["_pred"] == 1)],
        "true_positive": frame[(frame["_true"] == 1) & (frame["_pred"] == 1)],
    }

    analysis: dict = {}
    for bucket_name, rows in buckets.items():
        entry: dict = {"count": len(rows)}
        for feature in NAME_FEATURES_FOR_ERROR_ANALYSIS:
            if feature in rows.columns:
                entry[f"{feature}_rate"] = round(float(rows[feature].mean()), 3)
        entry["example_column_names"] = (
            rows["column_name"].head(ERROR_ANALYSIS_SAMPLE_SIZE).tolist()
        )
        entry["most_common_column_names"] = rows["column_name"].value_counts().head(10).to_dict()
        analysis[bucket_name] = entry

    false_positives = buckets["false_positive"]
    sibling_targets = [c for c in TARGET_COLUMNS if c != TARGET_COLUMN and c in frame.columns]
    if sibling_targets and len(false_positives) > 0:
        is_other_key = false_positives[sibling_targets].to_numpy().max(axis=1) == 1
        analysis["false_positive"]["sibling_target_rate"] = round(
            float(is_other_key.mean()),
            3,
        )

    return analysis


def print_error_analysis(name: str, analysis: dict) -> None:
    print(f"\n------Out-Of-Fold Error Analysis: {name}------\n")
    for bucket_name, entry in analysis.items():
        print(f"{bucket_name}: n={entry['count']}")
        rates = {k: v for k, v in entry.items() if k.endswith("_rate")}
        print(f"  rates: {rates}")
        print(f"  examples: {entry['example_column_names'][:15]}")


def confusion_matrix_figure(matrix: np.ndarray, title: str) -> Figure:
    """Render a 2x2 confusion matrix as a labelled heatmap.

    Uses the Figure API rather than pyplot so no interactive backend is needed - this
    runs headless in CI and inside the Prefect container.
    """
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_xticks([0, 1], labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    ax.set_yticks([0, 1], labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def precision_recall_figure(y_true: pd.Series, y_proba: np.ndarray, title: str) -> Figure:
    """Render the out-of-fold precision/recall curve with the no-skill baseline."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    ax.plot(recall, precision)
    ax.axhline(float(y_true.mean()), linestyle="--", linewidth=1, label="no skill")
    ax.set_title(title)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


def extract_params(model: ClassifierMixin) -> dict:
    """Return a flat, loggable hyperparameter dict for a candidate.

    For a TunedThresholdClassifierCV we log the underlying fitted estimator's
    params plus the tuned decision threshold, instead of the wrapper's nested
    `estimator` object (which would stringify to an unreadable blob).
    """
    if isinstance(model, TunedThresholdClassifierCV):
        base = getattr(model, "estimator_", model.estimator)
        params = dict(base.get_params())
        params["tuned_threshold"] = float(model.best_threshold_)
        return params
    return dict(model.get_params())


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
    input_path: Path,
) -> str:
    """Log one candidate as its own MLflow run and return the run id.

    Metric naming, all three prefixes describing different things:
    - `cv_f1_*`   : mean/std/min/max over the five per-fold F1 scores. The spread is
                    the number that showed the old single-fold score was noise.
    - `oof_*`     : computed once over the pooled out-of-fold predictions. These are
                    the honest quality estimates - judge the model by them.
    - `train_*`   : in-sample on the full refit, logged purely as an overfitting tell
                    (a large oof/train gap).
    The old `test_*` prefix is gone because there is no single holdout any more.

    The model artifact is logged with the sklearn flavor (works for RandomForest,
    XGBClassifier and the TunedThresholdClassifierCV wrapper) so the serving side can
    call predict_proba on the raw model.
    """
    input_example = X.head(5).astype(float)
    model = candidate["model"]
    signature = infer_signature(input_example, model.predict(input_example))
    cv_result = candidate["cv"]

    with mlflow.start_run(run_name=candidate["name"]) as run:
        mlflow.log_params(extract_params(model))

        mlflow.log_param("rows", len(X))
        mlflow.log_param("cv_splits", N_SPLITS)
        mlflow.log_param("cv_grouped_by", "database")
        mlflow.log_param("search_n_iter", SEARCH_N_ITER)
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("alias", MODEL_ALIAS)
        mlflow.log_param("candidate", candidate["name"])

        mlflow.log_metric("cv_f1_mean", cv_result["cv_f1_mean"])
        mlflow.log_metric("cv_f1_std", cv_result["cv_f1_std"])
        mlflow.log_metric("cv_f1_min", cv_result["cv_f1_min"])
        mlflow.log_metric("cv_f1_max", cv_result["cv_f1_max"])
        for fold, fold_score in enumerate(cv_result["fold_f1"]):
            mlflow.log_metric("cv_f1_per_fold", fold_score, step=fold)
        for metric_name, value in candidate["cv_metrics"].items():
            mlflow.log_metric(f"oof_{metric_name}", value)
        for metric_name, value in candidate["train_metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", value)

        mlflow.set_tags(
            {
                "model_type": candidate["model_type"],
                "candidate": candidate["name"],
                "developer": "test",
                "dataset": "trino-train-metadata-statistics",
                "target_column": y.name,
                "n_features": len(X.columns),
                "n_samples": len(X),
                "evaluation": f"{N_SPLITS}-fold StratifiedGroupKFold on database",
            },
        )

        mlflow.log_dict({"features": list(X.columns)}, "features.json")
        mlflow.log_dict(
            {
                "fold_f1": cv_result["fold_f1"],
                "fold_thresholds": cv_result["fold_thresholds"],
                "oof_confusion_matrix": cv_result["oof_confusion_matrix"].tolist(),
                "oof_metrics": candidate["cv_metrics"],
                "train_metrics": candidate["train_metrics"],
            },
            "cross_validation.json",
        )
        mlflow.log_figure(
            confusion_matrix_figure(
                cv_result["oof_confusion_matrix"],
                f"{candidate['name']} (out-of-fold)",
            ),
            "confusion_matrix_oof.png",
        )
        mlflow.log_figure(
            precision_recall_figure(
                y,
                cv_result["oof_proba"],
                f"{candidate['name']} (out-of-fold)",
            ),
            "precision_recall_oof.png",
        )

        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            serialization_format="pickle",
            signature=signature,
            input_example=input_example,
        )

    print(
        f"Logged run {run.info.run_id} for '{candidate['name']}' "
        f"(cv F1={cv_result['cv_f1_mean']:.4f} +/- {cv_result['cv_f1_std']:.4f})",
    )
    return run.info.run_id


def select_best_candidate(candidates: list[dict]) -> dict:
    """The candidate with the highest CV mean F1.

    Selecting on the CV mean rather than a single fold's score is the whole point of
    the rewrite: on one fold the ranking is dominated by which databases happened to
    land in it.
    """
    return max(candidates, key=lambda c: c["cv"]["cv_f1_mean"])


def register_best_candidate(
    best: dict,
    run_id: str,
    model_name: str,
    alias: str,
) -> ModelVersion:
    """Register the given candidate's run and point the serving alias at it."""
    print("\n------MLflow Model Registration------")

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    print(
        f"Best candidate: '{best['name']}' with cv F1={best['cv']['cv_f1_mean']:.4f} "
        f"+/- {best['cv']['cv_f1_std']:.4f}",
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
    input_path, raw_df = load_data()
    df = one_hot_encode_column_type(raw_df)

    X, y, groups = build_feature_matrix(df)
    splits = make_grouped_splits(X, y, groups)

    candidates = train_all_candidates(X, y, groups, splits)
    print_candidate_summary(candidates)

    best = select_best_candidate(candidates)
    error_analysis = analyse_errors(raw_df, y, best["cv"]["oof_pred"])
    print_error_analysis(best["name"], error_analysis)

    client = MlflowClient()
    setup_experiment(client, EXPERIMENT_NAME)

    run_ids: dict[str, str] = {}
    for candidate in candidates:
        run_ids[candidate["name"]] = log_candidate_run(
            candidate=candidate,
            X=X,
            y=y,
            input_path=input_path,
        )

    # The error analysis belongs to the run that produced it, so it is attached to the
    # winner's run rather than logged as a loose file next to the experiment.
    with mlflow.start_run(run_id=run_ids[best["name"]]):
        mlflow.log_dict(error_analysis, "error_analysis_oof.json")

    register_best_candidate(
        best=best,
        run_id=run_ids[best["name"]],
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    main()
