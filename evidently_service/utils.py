"""This module contains utility functions for Evidently monitoring service."""

import hashlib
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import prometheus_client
import yaml
from evidently import BinaryClassification, DataDefinition, Dataset, Report
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


def build_data_definition(config: Mapping[str, Any]) -> DataDefinition:
    """
    Build Evidently DataDefinition
    """
    # The config mirrors Evidently's idea of column roles: numerical features,
    # categorical features, target, and prediction.
    colmap = config["column_mapping"]

    num_features = colmap["numerical_features"]
    cat_features = colmap["categorical_features"]
    prediction = colmap["prediction"]

    # Define the schema Evidently uses when comparing reference and current data.
    # The prediction is monitored as a numerical column because the model returns
    # a continuous duration estimate.
    data_definition = DataDefinition(
        numerical_columns=[*num_features, prediction],
        categorical_columns=cat_features,
    )

    return data_definition


def create_report(_config: Mapping[str, Any]) -> Report:
    """
    Create Evidently Report based on configuration.

    Args:
        _config: Configuration dict (reserved for future use)
    """
    # Keep the starter report intentionally small: one prediction drift metric,
    # one count of drifted columns, and one missing-value check for predictions.
    report = Report(
        metrics=[
            cast("Any", ValueDrift)(column="prediction"),
            cast("Any", DriftedColumnsCount)(),
            cast("Any", MissingValueCount)(column="prediction"),
            cast("Any", ValueDrift)(column="trip_distance"),
        ],
    )
    logging.info("Evidently Report initialised.")
    return report


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
        dataset_name: str name of dataset (e.g. 'green_taxi_data')
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
) -> None:
    """
    Run the Evidently report, save HTML, extract metrics,
    and update Prometheus gauges.
    """
    logging.info("Running Evidently report...")

    # Convert the live pandas DataFrame into an Evidently Dataset with the same
    # column role definition used by the reference data.
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)

    # Run the comparison and save the rich HTML report for you to inspect.
    snapshot = report.run(reference_data=reference_data, current_data=current_dataset)
    snapshot.save_html("latest_report.html")

    # Export the report's numeric outputs to Prometheus for dashboards.
    metrics = extract_metrics_from_snapshot(snapshot)
    update_prometheus_metrics(metrics, gauge_store, dataset_name)

    logging.info(f"Report updated and {len(metrics)} metrics exported to Prometheus.")


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


def build_classification_data_definition(config: Mapping[str, Any]) -> DataDefinition:
    """
    Build the DataDefinition for the classification-quality report.

    Only the three scoring columns are declared. Features are deliberately left
    out: the metrics below do not read them, and declaring them would couple this
    track to the model's feature schema for no benefit.
    """
    clf = config["classification"]

    return DataDefinition(
        classification=[
            BinaryClassification(
                target=clf["target"],
                prediction_labels=clf["prediction"],
                prediction_probas=clf["prediction_proba"],
            ),
        ],
    )


def create_classification_report(_config: Mapping[str, Any]) -> Report:
    """
    Create the classification-quality Report.

    LogLoss is the only metric here that needs probabilities rather than hard
    labels; the rest are computed from the label column.
    """
    report = Report(
        metrics=[
            cast("Any", F1Score)(),
            cast("Any", Precision)(),
            cast("Any", Recall)(),
            cast("Any", Accuracy)(),
            cast("Any", LogLoss)(),
        ],
    )
    logging.info("Evidently classification report initialised.")
    return report


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
    """
    dataset = Dataset.from_pandas(scored_data, data_definition=data_definition)
    snapshot = report.run(reference_data=None, current_data=dataset)

    if save_html:
        snapshot.save_html("latest_classification_report.html")

    metrics = extract_metrics_from_snapshot(snapshot)
    update_classification_metrics(metrics, gauge_store, dataset_name, series_type)

    logging.info(
        f"Classification report ({series_type}) updated: "
        f"{len(metrics)} metrics exported for {len(scored_data)} rows.",
    )
    return metrics


def compute_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of the reference DataFrame."""
    # Hashing the reference data gives you a quick way to confirm that the
    # expected baseline dataset was loaded by the service.
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
