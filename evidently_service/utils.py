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
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift


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


def compute_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash of the reference DataFrame."""
    # Hashing the reference data gives you a quick way to confirm that the
    # expected baseline dataset was loaded by the service.
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
