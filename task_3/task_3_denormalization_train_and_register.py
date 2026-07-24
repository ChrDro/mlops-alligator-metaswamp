"""
In this notebook the prediction of the normalform key will be done.
We start with a lightGBM model.
"""

import os
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


df = pd.DataFrame()
RSEED = 42

# DEFAULT_INPUT_PATH = Path("evidently_service/green_taxi_data/reference.csv")
# DEFAULT_MODEL_NAME = "green-taxi-ride-duration"
# DEFAULT_ALIAS = "production"
# Read .env into a dict WITHOUT mutating os.environ. Its values target the Docker network
# (e.g. MLFLOW_TRACKING_URI=http://mlflow:5000, endpoints at minio:9000) and are wrong for
# a host-run script — we only want the MinIO credentials out of it.
_env = dotenv_values()

# The server is published on the host at localhost:5000 (the Docker name "mlflow" does not
# resolve here). Honor an explicit shell override, otherwise default to localhost.
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

# MLflow logs run artifacts (features.json, the model) directly to MinIO because the run's
# artifact root is s3://mlflow/. On the host this client must carry the MinIO endpoint +
# credentials itself, or boto3 falls back to real AWS S3 and fails with InvalidAccessKeyId.
# Endpoint is localhost:9000 (published port), NOT minio:9000 (Docker-network only).
_minio_user = _env.get("MINIO_ROOT_USER")
_minio_password = _env.get("MINIO_ROOT_PASSWORD")
if _minio_user and _minio_password:
    os.environ["AWS_ACCESS_KEY_ID"] = _minio_user
    os.environ["AWS_SECRET_ACCESS_KEY"] = _minio_password
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")


def wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int,
) -> ModelVersion:
    """Wait until the registered model version is ready to serve."""
    # MLflow registration can finish asynchronously depending on the backend.
    # Polling here keeps the local workflow predictable before the API tries
    # to resolve the production alias.
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


def load_data() -> tuple[str, pd.DataFrame]:
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
    pd.DataFrame,
    pd.DataFrame,
]:
    print("\n------Train Test Split------")

    columns_to_drop = [
        "database",
        "schema",
        "table_name",
        "column_name",
        "target_normal_form",
        "table_contains_1nf_violation",
    ]

    X = df.drop(columns=columns_to_drop)

    y = df["target_normal_form"].astype(int)

    print(f"Feature count: {X.shape[1]}")

    groups = df["table_name"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # df_train = pd.concat([X_train, y_train], axis=1)
    # df_test = pd.concat([X_test, y_test], axis=1)

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


def print_denormlization_target_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    print("\n------Target Distribution------")
    print(y_train.value_counts())
    print(y_test.value_counts())


def predict_light_gbm(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, np.ndarray]:
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


def create_summary_train_set(
    y_train: pd.DataFrame,
    y_pred_lgbm_train: np.ndarray,
    y_pred_proba_lgbm_train: np.ndarray,
) -> pd.DataFrame:
    """
    This function creates a pandas dataframe with the following score for each model trained
    on the train set:
    - precision score
    - recall score
    - f1-score
    - accuracy score
    - average precision score (precision area under curve)
    """

    print("\n------Evaluation Train Set------")

    metrics_data_train = {
        "Precision": [
            precision_score(y_train, y_pred_lgbm_train, average="weighted"),
        ],
        "Recall": [
            recall_score(y_train, y_pred_lgbm_train, average="weighted"),
        ],
        "f1_score": [
            f1_score(y_train, y_pred_lgbm_train, average="weighted"),
        ],
        "accuracy": [
            accuracy_score(y_train, y_pred_lgbm_train),
        ],
        "pr_auc": [
            average_precision_score(y_train, y_pred_proba_lgbm_train, average="weighted"),
        ],
    }

    df_metrics_train = pd.DataFrame(
        metrics_data_train,
        index=["Light_GBM_Train"],
    )

    return df_metrics_train


def create_summary_test_set(
    y_test: pd.DataFrame,
    y_pred_lgbm_test: np.ndarray,
    y_pred_proba_lgbm_test: np.ndarray,
) -> pd.DataFrame:
    """
    This function creates a pandas dataframe with the following score for each model trained
    on the test set:
    - precision score
    - recall score
    - f1-score
    - accuracy score
    - average precision score (precision area under curve)
    """

    print("\n------Evaluation Test Set------")

    metrics_data_test = {
        "Precision": [
            precision_score(y_test, y_pred_lgbm_test, average="weighted"),
        ],
        "Recall": [
            recall_score(y_test, y_pred_lgbm_test, average="weighted"),
        ],
        "f1_score": [
            f1_score(y_test, y_pred_lgbm_test, average="weighted"),
        ],
        "accuracy": [
            accuracy_score(y_test, y_pred_lgbm_test),
        ],
        "pr_auc": [
            average_precision_score(y_test, y_pred_proba_lgbm_test, average="weighted"),
        ],
    }

    df_metrics_test = pd.DataFrame(
        metrics_data_test,
        index=["Light_GBM_Test"],
    )

    return df_metrics_test


def print_evaluation_train_set(
    df_metrics_train: pd.DataFrame,
    y_train: pd.DataFrame,
    y_pred_lgbm_train: np.ndarray,
) -> None:

    print("------Evaluation Table Train Set------\n")
    print(df_metrics_train.head(10))

    print("\n------Confusion Matrix------\n")

    print("Light_GBM_Train:")
    print(confusion_matrix(y_train, y_pred_lgbm_train))


def print_evaluation_test_set(
    df_metrics_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_pred_lgbm_test: np.ndarray,
) -> None:

    print("\n------Evaluation Table Test Set------\n")
    print(df_metrics_test.head(10))

    print("\n------Confusion Matrix------\n")

    print("Light_GBM_Test:")
    print(confusion_matrix(y_test, y_pred_lgbm_test))


def register_model_to_mlflow(
    model_lgbm: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    input_path: Path,
    train_f1_score: float,
    test_f1_score: float,
    model_name: str,
    alias: str,
) -> ModelVersion:
    """
    Register trained model to MLflow with logging of parameters, metrics, and artifacts.
    Args:
        model_lgbm: Trained LightGBM model
        X: Full feature DataFrame before train/test split
        y: Full target Series before train/test split
        X_train: Training features
        X_test: Test features
        input_path: Path to training data
        train_f1_score: F1 score on training set
        test_f1_score: F1 score on test set
        model_name: Name for model registration
        alias: Alias for model version (e.g., 'dev', 'production')
    Returns:
        model_version: Registered model version object
    """
    print("\n------MLflow Model Registration------")

    input_example = X_train.head(5).astype(float)
    signature = infer_signature(input_example, model_lgbm.predict(input_example))

    client = MlflowClient()

    # Create or get existing experiment
    experiment_name = "denormalization_model_training"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="denormalization_model_training") as run:
        # Log a few training details so the bootstrap run is easy to inspect
        # in MLflow and understand where the registered model came from.
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("holdout_rows", len(X_test))
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("alias", alias)
        mlflow.log_metric("train_f1_score", train_f1_score)
        mlflow.log_metric("test_f1_score", test_f1_score)

        # Log metadata as tags
        mlflow.set_tags(
            {
                "model_type": "light_gbm_classifier",
                "developer": "test",
                "dataset": "generated-denormalization-statistics",
                "target_column": y.name,  # Uses the Series name
                "n_features": len(X.columns),
                "n_samples": len(X),
            },
        )

        # Log feature names as a dict parameter (better for programmatic access)
        mlflow.log_dict(
            {"features": list(X.columns)},
            "features.json",
        )

        mlflow.sklearn.log_model(
            model_lgbm,
            name="denormalization_model",
            serialization_format="pickle",
            signature=signature,
            input_example=input_example,
        )
        model_uri = f"runs:/{run.info.run_id}/denormalization_model"

    print(f"Logged run {run.info.run_id}")
    print(f"Registering {model_uri} as {model_name}")

    # Register the run artifact as a named model, then point the alias used by
    # the API at the new version.
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
    print_denormlization_target_distribution(y_train, y_test)

    # Predict LightGBM Model without hyperparameter search
    # print("\n------Predict LightGBM without hyperparameter search------\n")
    (
        model_lgbm,
        y_pred_lgbm_train,
        y_pred_lgbm_test,
        y_pred_proba_lgbm_train,
        y_pred_proba_lgbm_test,
    ) = predict_light_gbm(RSEED, X_train, y_train, X_test)

    # Evaluation
    df_metrics_train = create_summary_train_set(
        y_train,
        y_pred_lgbm_train,
        y_pred_proba_lgbm_train,
    )

    df_metrics_test = create_summary_test_set(
        y_test,
        y_pred_lgbm_test,
        y_pred_proba_lgbm_test,
    )

    print_evaluation_train_set(
        df_metrics_train,
        y_train,
        y_pred_lgbm_train,
    )

    print_evaluation_test_set(
        df_metrics_test,
        y_test,
        y_pred_lgbm_test,
    )

    train_f1_score = df_metrics_train.loc["Light_GBM_Train", "f1_score"]
    test_f1_score = df_metrics_test.loc["Light_GBM_Test", "f1_score"]

    # print(type(train_f1_score))
    print(train_f1_score)
    # print(type(test_f1_score))
    print(test_f1_score)

    model_name = "denormalization_model"
    alias = "dev"

    # Register model to MLflow
    register_model_to_mlflow(
        model_lgbm=model_lgbm,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        input_path=input_path,
        train_f1_score=train_f1_score,
        test_f1_score=test_f1_score,
        model_name=model_name,
        alias=alias,
    )


if __name__ == "__main__":
    main()
