"""
Train several composite-primary-key classifiers, log each as its own MLflow run,
then register the best one (by holdout F1) and point the serving alias at it.

Candidates (no baseline):
- XGBoost (defaults, scale_pos_weight)
- Random forest (defaults, balanced class weight)
- XGBoost (randomized search + tuned decision threshold)
- Random forest (randomized search + tuned decision threshold)

All candidates are logged with the mlflow.sklearn flavor on purpose: the web
service resolves the model via mlflow.pyfunc and then calls
`get_raw_model().predict_proba(...)`. The sklearn flavor keeps the estimator
(and its predict_proba) intact for RandomForest, XGBClassifier and the
TunedThresholdClassifierCV wrapper; the xgboost flavor would hand back a Booster
with no predict_proba.

For the tuned candidates we register the TunedThresholdClassifierCV itself (not
the raw best_estimator_), so the served model applies the same decision
threshold that produced the reported metrics.
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
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    TunedThresholdClassifierCV,
)


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

MODEL_ARTIFACT_NAME = "composite_pk_model"
MODEL_NAME = "composite_pk_model"
MODEL_ALIAS = "dev"


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
    pd.Series,
    pd.Series,
]:
    print("\n------Train Test Split------")

    columns_to_drop = [
        "database",
        "schema",
        "table_name",
        "column_name",
        "min_value",
        "max_value",
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

    y = df["composite_pk_target"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Total groups (tables): {groups.nunique()}")
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    train_cpk_rate = round(y_train.mean() * 100, 1)
    test_cpk_rate = round(y_test.mean() * 100, 1)
    print(f"Train composite pk rate: {train_cpk_rate}% | Test composite pk rate: {test_cpk_rate}%")
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


def print_composite_pk_target_distribution(y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
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


def predict_random_forest(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict Random Forest Model------")
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


def predict_xg_boost_randomized_search(
    RSEED: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[TunedThresholdClassifierCV, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict XGBoost Model with hyperparameter search------")
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)

    param_distributions_xgb = {
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": list(range(3, 11)),
        "min_child_weight": list(range(1, 15)),
        "gamma": [0, 0.05, 0.1, 0.2, 0.5],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.05, 0.1],
        "reg_lambda": [0.5, 1, 1.5, 2],
    }

    model_xgb_randomized = xgb.XGBClassifier(scale_pos_weight=spw, random_state=RSEED)

    randomized_search_xgb = RandomizedSearchCV(
        estimator=model_xgb_randomized,
        param_distributions=param_distributions_xgb,
        n_jobs=-1,
        n_iter=100,
        cv=3,
        verbose=1,
        scoring="average_precision",
        random_state=RSEED,
    )
    randomized_search_xgb.fit(X_train, y_train)

    print("------Best Score:------")
    print("Best score: ", randomized_search_xgb.best_score_)
    print("------Best Hyperparameters:------")
    print(randomized_search_xgb.best_params_)

    tuned_xgb = TunedThresholdClassifierCV(
        randomized_search_xgb.best_estimator_,
        scoring="f1",
        cv=5,
    )
    tuned_xgb.fit(X_train, y_train)
    print(f"Tuned threshold: {tuned_xgb.best_threshold_:.3f}")

    y_pred_train = tuned_xgb.predict(X_train)
    y_pred_test = tuned_xgb.predict(X_test)
    y_pred_proba_train = tuned_xgb.predict_proba(X_train)[:, 1]
    y_pred_proba_test = tuned_xgb.predict_proba(X_test)[:, 1]

    return (
        tuned_xgb,
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
) -> tuple[TunedThresholdClassifierCV, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n------Predict Random Forest Model with hyperparameter search------")

    param_distributions_rf = {
        "max_depth": list(range(5, 21)),
        "n_estimators": list(range(25, 501, 25)),
        "max_features": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        "min_samples_leaf": list(range(1, 11)),
        "min_samples_split": list(range(2, 21)),
    }

    model_rf_randomized = RandomForestClassifier(class_weight="balanced", random_state=RSEED)

    randomized_search_rf = RandomizedSearchCV(
        model_rf_randomized,
        param_distributions_rf,
        n_jobs=-1,
        n_iter=100,
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

    tuned_rf = TunedThresholdClassifierCV(
        randomized_search_rf.best_estimator_,
        scoring="f1",
        cv=5,
    )
    tuned_rf.fit(X_train, y_train)
    print(f"Tuned threshold: {tuned_rf.best_threshold_:.3f}")

    y_pred_train = tuned_rf.predict(X_train)
    y_pred_test = tuned_rf.predict(X_test)
    y_pred_proba_train = tuned_rf.predict_proba(X_train)[:, 1]
    y_pred_proba_test = tuned_rf.predict_proba(X_test)[:, 1]

    return (
        tuned_rf,
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
) -> list[dict]:
    """Train every candidate model and package each with its train/test metrics.

    The randomized-search candidates return a TunedThresholdClassifierCV; that
    tuned estimator is what gets scored and (if best) registered, so the served
    model applies the same threshold that produced the reported metrics.
    """
    print("\n------Training candidate models------")

    candidates: list[dict] = []

    model_xgb, *xgb_preds = predict_xg_boost(RSEED, X_train, y_train, X_test)
    candidates.append(
        _make_candidate("xgboost", "xgboost_classifier", model_xgb, y_train, y_test, xgb_preds),
    )

    model_rf, *rf_preds = predict_random_forest(RSEED, X_train, y_train, X_test)
    candidates.append(
        _make_candidate(
            "random_forest",
            "random_forest_classifier",
            model_rf,
            y_train,
            y_test,
            rf_preds,
        ),
    )

    tuned_xgb, *xgb_rs_preds = predict_xg_boost_randomized_search(RSEED, X_train, y_train, X_test)
    candidates.append(
        _make_candidate(
            "xgboost_randomized_search",
            "xgboost_classifier_tuned_threshold",
            tuned_xgb,
            y_train,
            y_test,
            xgb_rs_preds,
        ),
    )

    tuned_rf, *rf_rs_preds = predict_random_forest_randomized_search(
        RSEED,
        X_train,
        y_train,
        X_test,
    )
    candidates.append(
        _make_candidate(
            "random_forest_randomized_search",
            "random_forest_classifier_tuned_threshold",
            tuned_rf,
            y_train,
            y_test,
            rf_rs_preds,
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
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    input_path: Path,
) -> str:
    """Log one candidate as its own MLflow run and return the run id.

    The model artifact is logged with the sklearn flavor (works for RandomForest,
    XGBClassifier and the TunedThresholdClassifierCV wrapper) so the serving side
    can call predict_proba on the raw model.
    """
    input_example = X_train.head(5).astype(float)
    model = candidate["model"]
    signature = infer_signature(input_example, model.predict(input_example))

    with mlflow.start_run(run_name=candidate["name"]) as run:
        mlflow.log_params(extract_params(model))

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
    print_composite_pk_target_distribution(y_train, y_test)

    # Train all candidate models and score them.
    candidates = train_all_candidates(RSEED, X_train, y_train, X_test, y_test)
    print_candidate_summary(candidates)

    # Log every candidate as its own run under one experiment.
    client = MlflowClient()
    setup_experiment(client, "composite_pk_model_training")

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
