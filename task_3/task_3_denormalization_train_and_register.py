"""
Train several normal-form (denormalization) classifiers, log each as its own
MLflow run, then register the best one (by holdout weighted-F1) and point the
serving alias at it.

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

import os
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
from sklearn.base import ClassifierMixin
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


df = pd.DataFrame()
RSEED = 42

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
    print("\n------Data Loading------")
    current_pwd = Path.cwd().resolve()
    train_data_path = "data/nf_test_analyse.csv"
    input_path = current_pwd / train_data_path
    df = pd.read_csv(input_path)
    print(df.head())
    return input_path, df


def one_hot_encode_column_type(df: pd.DataFrame) -> pd.DataFrame:
    print("\n------Column Preview and One-Hot Encoding of 'column_type'------")
    df = pd.get_dummies(df, columns=["column_type"], drop_first=True)
    print(df.columns)
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

    # Identifiers, the target, and the leakage guard. 50 -> 44 features.
    columns_to_drop = [
        "database",
        "schema",
        "table_name",
        "column_name",
        "target_normal_form",
        "table_contains_1nf_violation",
    ]

    # Phase 0 of TASK_3_PLAN.md: 21 features carry no information of their own. Dropping
    # them left the holdout weighted-F1 unchanged (0.9935 -> 0.9939), so this buys
    # interpretable feature importances, not accuracy. 44 -> 29 features.
    #
    # NOT dropped: the table-level aggregates (table_ratio_1nf_violations,
    # table_has_partial_dependency, table_avg_unique_ratio, table_ratio_composite_key_cols).
    # They look derivable via groupby, but the model sees ONE row per call and cannot
    # aggregate - dropping them collapses F1 to 0.7710.
    redundant_columns_to_drop = [
        # Constant in nf_test_analyse.csv (no NULLs and no name matches anywhere), so
        # they carry zero bits. See TASK_3_PLAN.md 1.1 and 1.5.
        "null_count",
        "null_ratio",
        "is_non_null",
        "table_avg_null_ratio",
        "name_contains_key",
        "name_contains_table_name",
        "name_is_singular_table_id",
        # Bit-for-bit identical to another column in this dataset.
        "count",  # == table_row_count
        "null_ratio_rank",  # == ordinal_position (no NULLs -> only the tiebreaker ranks)
        "table_non_null_column_count",  # == table_column_count (dito)
        # Threshold derivations of a single value in the SAME row - a tree can split on
        # the source column itself, so these add nothing.
        "is_first_column",  # == (ordinal_position == 1)
        "is_least_null_in_table",  # == (null_ratio_rank == 1)
        "is_unique",  # == (unique_ratio == 1 and null_count == 0)
        "table_has_unique_column",  # == (table_unique_column_count > 0)
        "table_has_no_single_pk_candidate",  # == 1 - table_has_unique_column
        # Differences of two features that both stay in the set.
        "other_unique_columns_in_table",  # == table_unique_column_count - is_unique
        "other_near_unique_columns_in_table",  # == table_near_unique_column_count - ...
        # Raw value whose ratio is kept; the ratio is the informative half because trees
        # cannot divide.
        "ordinal_position",  # -> relative_ordinal_position
        "number_unique_values",  # -> unique_ratio
        "table_unique_column_count",  # -> table_ratio_of_pk_candidates
        "unique_ratio_relative_to_max",  # == unique_ratio / table_max_unique_ratio
        # "table_ratio_1nf_violations",
        # "table_has_partial_dependency",
    ]

    X = df.drop(columns=columns_to_drop + redundant_columns_to_drop)

    y = df["target_normal_form"].astype(int)

    print(f"Feature count: {X.shape[1]}")

    groups = df["table_name"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Total groups (tables): {groups.nunique()}")
    train_tables = groups.iloc[train_idx].nunique()
    test_tables = groups.iloc[test_idx].nunique()
    print(f"Tables in train: {train_tables} | in test: {test_tables}")
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
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("alias", MODEL_ALIAS)
        mlflow.log_param("candidate", candidate["name"])

        for metric_name, value in candidate["train_metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in candidate["test_metrics"].items():
            mlflow.log_metric(f"test_{metric_name}", value)

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


def register_best_candidate(
    candidates: list[dict],
    run_ids: dict[str, str],
    model_name: str,
    alias: str,
) -> ModelVersion:
    """Register the highest test weighted-F1 candidate and point the alias at it."""
    print("\n------MLflow Model Registration------")

    best = max(candidates, key=lambda c: c["test_metrics"]["f1_score"])
    best_run_id = run_ids[best["name"]]
    best_f1 = best["test_metrics"]["f1_score"]
    model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_NAME}"

    print(f"Best candidate: '{best['name']}' with test weighted-F1={best_f1:.4f}")
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
        )

    # Register the best candidate (by holdout weighted-F1) and move the serving alias.
    register_best_candidate(
        candidates=candidates,
        run_ids=run_ids,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    main()
