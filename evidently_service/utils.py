"""This module contains utility functions for Evidently monitoring service."""

import hashlib
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import prometheus_client
import yaml
from evidently import (
    BinaryClassification,
    DataDefinition,
    Dataset,
    MulticlassClassification,
    Report,
)
from evidently.metrics import (
    Accuracy,
    DriftedColumnsCount,
    F1Score,
    LogLoss,
    MissingValueCount,
    Precision,
    Recall,
    ValueDrift,
)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load Evidently configuration from YAML.
    """
    # Fail early with a clear error if the Docker image or local run is missing
    # the monitoring configuration file.
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        msg = f"Configuration file {config_path} not found."
        logging.error(msg)
        raise FileNotFoundError(msg)

    # yaml.safe_load parses the simple config without executing arbitrary YAML
    # tags.
    with config_path_obj.open("rb") as file:
        config = yaml.safe_load(file)

    logging.info(f"Configuration loaded from {config_path}.")
    return cast("dict[str, Any]", config)


# Columns present in a reference file that describe the outcome rather than the
# input, so they are never treated as features.
NON_FEATURE_COLUMNS = frozenset({"prediction", "prediction_proba", "probability"})


def infer_column_roles(
    reference_df: pd.DataFrame,
    target: str,
    force_categorical: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """
    Split a reference file's feature columns into numerical and categorical.

    The reference is generated from the Pydantic request models, so its columns are
    exactly the served feature set. Inferring the split from it keeps this in step
    with the API automatically, instead of repeating ~35 column names per model in
    config.yaml and letting the two drift apart.

    Rule: booleans and two-valued numerics are categorical (binary flags, one-hot
    dummies); anything else numeric is numerical (counts, ratios, ranks).

    Returns:
        (numerical_columns, categorical_columns), both sorted for stable logging.
    """
    numerical: list[str] = []
    categorical: list[str] = []
    forced = set(force_categorical)

    for column in reference_df.columns:
        if column == target or column in NON_FEATURE_COLUMNS:
            continue

        series = reference_df[column]
        if (
            column in forced
            or series.dtype == bool
            or (pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) <= 2)
        ):
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numerical.append(column)
        else:
            # Non-numeric, non-bool: an identity or free-text column that should not
            # have reached the reference file. Skipping beats letting Evidently pick
            # a drift test for it.
            logging.warning(f"Skipping non-numeric reference column {column!r}")

    return sorted(numerical), sorted(categorical)


def build_data_definition(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    prediction: str,
    prediction_type: str = "numerical",
) -> DataDefinition:
    """
    Build the DataDefinition Evidently uses to compare reference and current data.

    Where the prediction column belongs depends on what the model outputs. These key
    models return a 0/1 label, so it is categorical; a regression model's output
    would be numerical. Getting this wrong silently applies the wrong drift test
    rather than failing, so it comes from config instead of being guessed.
    """
    num = list(numerical_columns)
    cat = list(categorical_columns)

    if prediction_type == "categorical":
        cat.append(prediction)
    else:
        num.append(prediction)

    return DataDefinition(numerical_columns=num, categorical_columns=cat)


def create_report(
    drift_columns: Sequence[str],
    prediction: str,
    known_columns: Iterable[str],
) -> Report:
    """
    Create the drift Report for one track.

    DriftedColumnsCount summarises every column in the DataDefinition; the columns in
    drift_columns additionally get their own ValueDrift series so they can be tracked
    individually on a dashboard.
    """
    known = set(known_columns) | {prediction}
    unknown = [c for c in drift_columns if c not in known]
    if unknown:
        # Evidently would otherwise raise mid-report on the first live window, which
        # surfaces as a 500 on /iterate long after the config was edited.
        msg = (
            f"drift_columns lists {unknown}, which are not columns of this track's "
            f"reference dataset. Fix the names or remove them from drift_columns."
        )
        raise ValueError(msg)

    metrics: list[Any] = [
        cast("Any", DriftedColumnsCount)(),
        cast("Any", MissingValueCount)(column=prediction),
    ]
    metrics.extend(cast("Any", ValueDrift)(column=column) for column in drift_columns)

    return Report(metrics=metrics)


def extract_metrics_from_snapshot(snapshot: Any) -> dict[str, Any]:
    """
    Convert snapshot.dict()['metrics'] into a flat dict of metric_name to value.
    If a column name is present (e.g. column=prediction), include it in the key.
    """
    metrics_list = snapshot.dict().get("metrics", [])
    results: dict[str, Any] = {}

    for metric in metrics_list:
        # Evidently stores each metric result with an id like
        # ValueDrift(column=prediction). The Prometheus exporter needs a stable,
        # flat name derived from that id.
        metric_id = metric.get("metric_id", "")
        value = metric.get("value")

        # Base metric name, e.g. "ValueDrift".
        base_name = metric_id.split("(")[0]

        # Extract column name if present: (column=prediction).
        col_match = re.search(r"column=([\w\d_]+)", metric_id)
        column_suffix = f"_{col_match.group(1)}" if col_match else ""

        # Build prefix for this metric.
        metric_prefix = f"{base_name}{column_suffix}"

        # Some Evidently metric values are already numeric; others are nested
        # dictionaries. Flatten dictionary values so each number can become one
        # Prometheus gauge.
        if isinstance(value, dict):
            for key, val in value.items():
                metric_key = f"{metric_prefix}_{key}"
                try:
                    results[metric_key] = float(val)
                except (ValueError, TypeError):
                    results[metric_key] = val
        else:
            try:
                results[metric_prefix] = float(value)
            except (ValueError, TypeError):
                results[metric_prefix] = value

    return results


def update_prometheus_metrics(
    metrics_dict: Mapping[str, Any],
    gauge_store: dict[str, prometheus_client.Gauge],
    dataset_name: str = "default_dataset",
) -> None:
    """
    Update or register Prometheus Gauges dynamically with dataset_name label.

    Args:
        metrics_dict: dict of {metric_name: value}
        gauge_store: dict to cache prometheus_client.Gauge objects
        dataset_name: the monitoring track (e.g. 'pk_columns'), which becomes the
            dataset_name label and is what the Grafana Model variable switches on
    """
    for name, value in metrics_dict.items():
        # Prometheus metric names are lowercase and prefixed so they are easy to
        # search from the Prometheus query UI.
        gauge_name = f"evidently_{name.lower()}"

        # Create the gauge if it doesn't exist yet.
        if name not in gauge_store:
            gauge_store[name] = prometheus_client.Gauge(
                gauge_name,
                f"Evidently metric {name}",
                labelnames=["dataset_name"],
            )
            logging.info(f"Registered Prometheus gauge: {gauge_name}")

        # Always set the latest value for this dataset. Grafana panels read the
        # most recently scraped values from Prometheus.
        gauge_store[name].labels(dataset_name=dataset_name).set(float(value))


def run_evidently(
    reference_data: Any,
    current_data: pd.DataFrame,
    data_definition: DataDefinition,
    report: Report,
    gauge_store: dict[str, prometheus_client.Gauge],
    dataset_name: str = "default_dataset",
    html_path: str = "latest_report.html",
) -> None:
    """
    Run the Evidently report, save HTML, extract metrics,
    and update Prometheus gauges.

    Args:
        html_path: Per-track filename. With several tracks sharing one process, a
            single fixed filename would mean each track overwrote the others'
            report and /report/<track> served whichever ran last.
    """
    # Convert the live pandas DataFrame into an Evidently Dataset with the same
    # column role definition used by the reference data.
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)

    # Run the comparison and save the rich HTML report for you to inspect.
    snapshot = report.run(reference_data=reference_data, current_data=current_dataset)
    snapshot.save_html(html_path)

    # Export the report's numeric outputs to Prometheus for dashboards.
    metrics = extract_metrics_from_snapshot(snapshot)
    update_prometheus_metrics(metrics, gauge_store, dataset_name)

    logging.info(f"[{dataset_name}] {len(metrics)} drift metrics exported")


# ------------------------------------------------------------------------------
# Classification quality track
#
# The drift track above needs no labels: it compares feature and prediction
# distributions. F1/precision/recall/log-loss are different - they score
# predictions against ground truth, so every event fed to this track must carry
# the true label. That is why this is a separate endpoint and a separate report
# rather than extra metrics on the drift report.
# ------------------------------------------------------------------------------


def positive_class_probability(prediction: int, probability: float) -> float:
    """
    Recover P(class=1) from the model API's ``probability`` field.

    /predict_pk returns ``probability`` as the confidence in the *predicted*
    class - ``max(predict_proba)``, see webservice/predict.py - not P(class=1).
    Feeding that to log-loss directly would score a confident 0-prediction as if
    it were a confident 1-prediction. For a binary argmax classifier the two are
    related exactly, so the true positive-class probability is recoverable:

        predicted 1 -> P(1) = probability
        predicted 0 -> P(1) = 1 - probability
    """
    return float(probability) if int(prediction) == 1 else 1.0 - float(probability)


def build_classification_data_definition(
    target: str,
    prediction: str,
    prediction_proba: str | None = None,
    multiclass: bool = False,
) -> DataDefinition:
    """
    Build the DataDefinition for one track's classification-quality report.

    Only the scoring columns are declared. Features are deliberately left out: the
    metrics do not read them, and declaring them would couple this track to the
    model's feature schema for no benefit.

    Args:
        prediction_proba: P(class=1) column. Binary tracks only - omitted for
            multiclass, where the API's single confidence value cannot be expanded
            into a full probability vector.
        multiclass: Use MulticlassClassification instead of BinaryClassification.
    """
    if multiclass:
        classification = MulticlassClassification(
            target=target,
            prediction_labels=prediction,
        )
    else:
        classification = BinaryClassification(
            target=target,
            prediction_labels=prediction,
            prediction_probas=prediction_proba,
        )

    return DataDefinition(classification=[classification])


def create_classification_report(multiclass: bool = False) -> Report:
    """
    Create a classification-quality Report.

    LogLoss is the only metric here needing probabilities rather than hard labels,
    and it is excluded for multiclass tracks: without a prediction_proba column
    Evidently raises "Cannot compute LogLoss: current LogLoss value is missing"
    and the whole report run fails, so it cannot simply be left in and ignored.

    For multiclass, F1/precision/recall are MACRO-averaged (verified against
    sklearn: Evidently's values match average="macro" exactly). Macro weights every
    class equally, so a rare normal form counts as much as a common one.
    """
    metrics: list[Any] = [
        cast("Any", F1Score)(),
        cast("Any", Precision)(),
        cast("Any", Recall)(),
        cast("Any", Accuracy)(),
    ]
    if not multiclass:
        metrics.append(cast("Any", LogLoss)())

    return Report(metrics=metrics)


def update_classification_metrics(
    metrics_dict: Mapping[str, Any],
    gauge_store: dict[str, prometheus_client.Gauge],
    dataset_name: str = "default_dataset",
    series_type: str = "current",
) -> None:
    """
    Export classification metrics as ``evidently_clf_*`` gauges.

    Carries a ``type`` label ("current" or "reference") so one panel can plot the
    live score against the baseline. Evidently's snapshot only ever reports the
    current window - passing reference_data does not add reference values to it -
    so the two series are produced by scoring the two datasets separately.

    The ``evidently_clf_`` prefix keeps these distinct from the drift track's
    ``evidently_*`` gauges, which carry a different label set. Re-registering one
    metric name under two label sets is an error in prometheus_client.
    """
    for name, value in metrics_dict.items():
        gauge_name = f"evidently_clf_{name.lower()}"

        if name not in gauge_store:
            gauge_store[name] = prometheus_client.Gauge(
                gauge_name,
                f"Evidently classification metric {name}",
                labelnames=["dataset_name", "type"],
            )
            logging.info(f"Registered Prometheus gauge: {gauge_name}")

        gauge_store[name].labels(dataset_name=dataset_name, type=series_type).set(float(value))


def run_classification_report(
    scored_data: pd.DataFrame,
    data_definition: DataDefinition,
    report: Report,
    gauge_store: dict[str, prometheus_client.Gauge],
    dataset_name: str = "default_dataset",
    series_type: str = "current",
    save_html: bool = True,
    html_path: str = "latest_classification_report.html",
) -> dict[str, Any]:
    """
    Score one labelled dataset and export the result to Prometheus.

    Args:
        scored_data: Rows carrying target, predicted label and positive-class
            probability. Used as Evidently's *current* dataset regardless of
            series_type - see update_classification_metrics for why.
        series_type: "current" for the live window, "reference" for the baseline.
        save_html: Skipped for the reference pass, which runs once at startup and
            would otherwise overwrite the live report file.
        html_path: Per-track filename, so tracks sharing this process do not
            overwrite each other's report.
    """
    dataset = Dataset.from_pandas(scored_data, data_definition=data_definition)
    snapshot = report.run(reference_data=None, current_data=dataset)

    if save_html:
        snapshot.save_html(html_path)

    metrics = extract_metrics_from_snapshot(snapshot)
    update_classification_metrics(metrics, gauge_store, dataset_name, series_type)

    logging.info(
        f"[{dataset_name}] classification ({series_type}): "
        f"{len(metrics)} metrics from {len(scored_data)} rows",
    )
    return metrics


def compute_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of the reference DataFrame."""
    # Hashing the reference data gives you a quick way to confirm that the
    # expected baseline dataset was loaded by the service.
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
