"""
Train several primary-key classifiers, log each as its own MLflow run, then
register the best one (by holdout F1) and point the serving alias at it.

Candidates:
- Random forest (defaults, balanced class weight)
- Random forest (randomized hyperparameter search)
- XGBoost (defaults, scale_pos_weight)
- XGBoost (randomized hyperparameter search)

All candidates are logged with the mlflow.sklearn flavor on purpose: the web
service resolves the model via mlflow.pyfunc and then calls
`get_raw_model().predict_proba(...)`. The sklearn flavor keeps the estimator
(and its predict_proba) intact for both RandomForest and XGBClassifier; the
xgboost flavor would hand back a Booster with no predict_proba.
"""

import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import dotenv_values
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold


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

MODEL_ARTIFACT_NAME = "pk_model"
MODEL_NAME = "pk_model"
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
            msg_model_registration_failed = (
                f"Model version {model_name} v{version} failed registration."
            )
            raise RuntimeError(
                msg_model_registration_failed,
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


def train_data_train_test_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print("\n------Train Test Split------")

    columns_to_drop = [
        "database",
        "schema",
        "table_name",
        "column_name",
        "min_value",
        "max_value",
        "null_count",
        "null_ratio",
        "is_non_null",
        "other_unique_columns_in_table",
        "other_near_unique_columns_in_table",
        "name_contains_key",
        "table_integer_column_count",
    ]

    groups = df["database"]

    X = df.drop(
        columns=[
            "pk_target",
            "composite_pk_target",
            "fk_target",
            "composite_fk_target",
            *columns_to_drop,
        ],
    )

    print(f"Feature count: {X.shape[1]}")

    print(X.info())

    y = df["pk_target"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Total groups (tables): {groups.nunique()}")
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    train_pk_rate = round(y_train.mean() * 100, 1)
    test_pk_rate = round(y_test.mean() * 100, 1)
    print(f"Train single pk rate: {train_pk_rate}% | Test single pk rate: {test_pk_rate}%")
    train_tables = groups.iloc[train_idx].nunique()
    test_tables = groups.iloc[test_idx].nunique()
    print(f"Tables in train: {train_tables} | in test: {test_tables}")

    return X, y, X_train, X_test, y_train, y_test


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


def print_pk_target_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    print("\n------Target Distribution------")
    print(y_train.value_counts())
    print(y_test.value_counts())


def predict_random_forest(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_rf = RandomForestClassifier(
        random_state=RSEED,
        class_weight="balanced",
    )
    model_rf.fit(X_train, y_train)

    y_pred_rf_train = model_rf.predict(X_train)
    y_pred_rf_test = model_rf.predict(X_test)
    y_pred_proba_rf_train = model_rf.predict_proba(X_train)[:, 1]
    y_pred_proba_rf_test = model_rf.predict_proba(X_test)[:, 1]

    return (
        model_rf,
        y_pred_rf_train,
        y_pred_rf_test,
        y_pred_proba_rf_train,
        y_pred_proba_rf_test,
    )


def predict_xg_boost(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict XGBoost Model------")
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)

    model_xgb = xgb.XGBClassifier(
        scale_pos_weight=spw,
        random_state=RSEED,
    )
    model_xgb.fit(X_train, y_train)

    y_pred_xgb_train = model_xgb.predict(X_train)
    y_pred_xgb_test = model_xgb.predict(X_test)
    y_pred_proba_xgb_train = model_xgb.predict_proba(X_train)[:, 1]
    y_pred_proba_xgb_test = model_xgb.predict_proba(X_test)[:, 1]

    return (
        model_xgb,
        y_pred_xgb_train,
        y_pred_xgb_test,
        y_pred_proba_xgb_train,
        y_pred_proba_xgb_test,
    )


def predict_xg_boost_randomized_search(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict XGBoost Model with hyperparameter search------")
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)

    param_distributions_xgb = {
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": list(range(3, 11)),
        "min_child_weight": list(range(1, 15)),
        "gamma": [0, 0.1, 0.5, 1, 1.5, 2, 3, 5],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1, 2, 5],
        "reg_lambda": [0.5, 1, 1.5, 2, 3, 5],
    }

    model_xgb_randomized = xgb.XGBClassifier(scale_pos_weight=spw, random_state=RSEED)

    randomized_search_xgb = RandomizedSearchCV(
        estimator=model_xgb_randomized,
        param_distributions=param_distributions_xgb,
        n_iter=50,
        cv=5,
        verbose=1,
        scoring="average_precision",
        random_state=RSEED,
    )
    randomized_search_xgb.fit(X_train, y_train)

    print("------Best Score:------")
    print("Best score: ", randomized_search_xgb.best_score_)
    print("------Best Hyperparameters:------")
    print(randomized_search_xgb.best_params_)

    best_estimator = randomized_search_xgb.best_estimator_

    y_pred_train = randomized_search_xgb.predict(X_train)
    y_pred_test = randomized_search_xgb.predict(X_test)
    y_pred_proba_train = randomized_search_xgb.predict_proba(X_train)[:, 1]
    y_pred_proba_test = randomized_search_xgb.predict_proba(X_test)[:, 1]

    return (
        best_estimator,
        y_pred_train,
        y_pred_test,
        y_pred_proba_train,
        y_pred_proba_test,
    )


def predict_random_forest_randomized_search(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict Random Forest Model with hyperparameter search------")

    param_distributions_rf = {
        "max_depth": list(range(3, 21)),
        "n_estimators": list(range(50, 501, 25)),
        "max_features": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        "min_samples_leaf": list(range(1, 11)),
        "min_samples_split": list(range(2, 21)),
    }

    model_rf_randomized = RandomForestClassifier(class_weight="balanced", random_state=RSEED)

    randomized_search_rf = RandomizedSearchCV(
        model_rf_randomized,
        param_distributions_rf,
        n_iter=50,
        cv=3,
        scoring="average_precision",
        verbose=1,
        random_state=RSEED,
    )
    randomized_search_rf.fit(X_train, y_train)

    print("------Best Hyperparameters:------")
    print(str(randomized_search_rf.best_params_))
    print("------Best Score:------")
    print("Best score is: " + str(randomized_search_rf.best_score_))

    best_estimator = randomized_search_rf.best_estimator_

    y_pred_train = randomized_search_rf.predict(X_train)
    y_pred_test = randomized_search_rf.predict(X_test)
    y_pred_proba_train = randomized_search_rf.predict_proba(X_train)[:, 1]
    y_pred_proba_test = randomized_search_rf.predict_proba(X_test)[:, 1]

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
    """Precision, recall, F1, accuracy, and PR-AUC for one set of predictions."""
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
    }


def train_all_candidates(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """Train every candidate model and package it with its train/test metrics.

    Each candidate is a dict: name, model_type, model, train_metrics, test_metrics,
    and the confusion matrices for console reporting.
    """
    print("\n------Training candidate models------")

    candidates: list[dict] = []

    (
        model_rf,
        y_pred_rf_train,
        y_pred_rf_test,
        y_pred_proba_rf_train,
        y_pred_proba_rf_test,
    ) = predict_random_forest(RSEED, X_train, y_train, X_test)
    candidates.append(
        {
            "name": "random_forest",
            "model_type": "random_forest_classifier",
            "model": model_rf,
            "train_metrics": compute_metrics(y_train, y_pred_rf_train, y_pred_proba_rf_train),
            "test_metrics": compute_metrics(y_test, y_pred_rf_test, y_pred_proba_rf_test),
            "train_confusion_matrix": confusion_matrix(y_train, y_pred_rf_train),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred_rf_test),
        },
    )

    (
        model_xgb,
        y_pred_xgb_train,
        y_pred_xgb_test,
        y_pred_proba_xgb_train,
        y_pred_proba_xgb_test,
    ) = predict_xg_boost(RSEED, X_train, y_train, X_test)
    candidates.append(
        {
            "name": "xgboost",
            "model_type": "xgboost_classifier",
            "model": model_xgb,
            "train_metrics": compute_metrics(y_train, y_pred_xgb_train, y_pred_proba_xgb_train),
            "test_metrics": compute_metrics(y_test, y_pred_xgb_test, y_pred_proba_xgb_test),
            "train_confusion_matrix": confusion_matrix(y_train, y_pred_xgb_train),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred_xgb_test),
        },
    )

    (
        best_rf,
        y_pred_rf_rs_train,
        y_pred_rf_rs_test,
        y_pred_proba_rf_rs_train,
        y_pred_proba_rf_rs_test,
    ) = predict_random_forest_randomized_search(RSEED, X_train, y_train, X_test)
    candidates.append(
        {
            "name": "random_forest_randomized_search",
            "model_type": "random_forest_classifier",
            "model": best_rf,
            "train_metrics": compute_metrics(
                y_train,
                y_pred_rf_rs_train,
                y_pred_proba_rf_rs_train,
            ),
            "test_metrics": compute_metrics(y_test, y_pred_rf_rs_test, y_pred_proba_rf_rs_test),
            "train_confusion_matrix": confusion_matrix(y_train, y_pred_rf_rs_train),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred_rf_rs_test),
        },
    )

    (
        best_xgb,
        y_pred_xgb_rs_train,
        y_pred_xgb_rs_test,
        y_pred_proba_xgb_rs_train,
        y_pred_proba_xgb_rs_test,
    ) = predict_xg_boost_randomized_search(RSEED, X_train, y_train, X_test)
    candidates.append(
        {
            "name": "xgboost_randomized_search",
            "model_type": "xgboost_classifier",
            "model": best_xgb,
            "train_metrics": compute_metrics(
                y_train,
                y_pred_xgb_rs_train,
                y_pred_proba_xgb_rs_train,
            ),
            "test_metrics": compute_metrics(y_test, y_pred_xgb_rs_test, y_pred_proba_xgb_rs_test),
            "train_confusion_matrix": confusion_matrix(y_train, y_pred_xgb_rs_train),
            "test_confusion_matrix": confusion_matrix(y_test, y_pred_xgb_rs_test),
        },
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

    The model artifact is logged with the sklearn flavor (works for both
    RandomForest and XGBClassifier) so the serving side can call predict_proba
    on the raw model.
    """
    input_example = X_train.head(5).astype(float)
    model = candidate["model"]
    signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run(run_name=candidate["name"]) as run:
        # Model hyperparameters (get_params captures the searched values for the
        # tuned candidates and the defaults for the rest).
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
                "dataset": "trino-train-metadata-statistics",
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
    print(f"Logged run {run.info.run_id} for '{candidate['name']}' (test F1={test_f1:.4f})")
    return run.info.run_id


def register_best_candidate(
    candidates: list[dict],
    run_ids: dict[str, str],
    model_name: str,
    alias: str,
) -> ModelVersion:
    """Register the highest test-F1 candidate and point the alias at it."""
    print("\n------MLflow Model Registration------")

    best = max(candidates, key=lambda c: c["test_metrics"]["f1_score"])
    best_run_id = run_ids[best["name"]]
    best_f1 = best["test_metrics"]["f1_score"]
    model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_NAME}"

    print(f"Best candidate: '{best['name']}' with test F1={best_f1:.4f}")
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
    ) = train_data_train_test_split(df)

    print_x_y_shape(X_train, X_test, y_train, y_test)
    print_pk_target_distribution(y_train, y_test)

    # Train all candidate models and score them.
    candidates = train_all_candidates(RSEED, X_train, y_train, X_test, y_test)
    print_candidate_summary(candidates)

    # Log every candidate as its own run under one experiment.
    client = MlflowClient()
    setup_experiment(client, "pk_model_training")

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

    # Register the best candidate (by holdout F1) and move the serving alias.
    register_best_candidate(
        candidates=candidates,
        run_ids=run_ids,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    main()
