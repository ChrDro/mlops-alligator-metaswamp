import logging
from pathlib import Path
from typing import Any

import pandas as pd
import prometheus_client
from flask import Flask, request, send_file
from flask.typing import ResponseReturnValue
from utils import (
    Dataset,
    build_data_definition,
    compute_hash,
    create_report,
    load_config,
    run_evidently,
)
from werkzeug.middleware.dispatcher import DispatcherMiddleware


# ------------------------------------------------------------------------------
# Flask + Prometheus setup
# ------------------------------------------------------------------------------

# Create a Flask application.
app = Flask(__name__)

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],  # Sends log messages to the console (stdout).
)

# Add Prometheus metrics endpoint.
# Create a WSGI application for serving Prometheus metrics.
prometheus_metrics_app = prometheus_client.make_wsgi_app()

# Combine the Flask app with the Prometheus metrics app using DispatcherMiddleware.
# This allows the Flask app to run alongside the Prometheus metrics endpoint.
# Any request to /metrics will be routed to the prometheus_metrics_app.
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": prometheus_metrics_app})


def init_evidently() -> None:
    """Load config, reference data and initialize Evidently report."""
    logging.info("Initializing Evidently monitoring service...")

    # Build paths from this file's location so the service works whether it is
    # launched from the repo root or from inside the Docker container.
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.yaml"
    config = load_config(config_path)
    data_definition = build_data_definition(config)

    # The config keeps the reference path relative to evidently_service/.
    # Convert it to an absolute path before pandas reads it.
    reference_path = config["service"]["reference_path"]
    if not Path(reference_path).is_absolute():
        reference_path = base_dir / reference_path
    reference_df = pd.read_csv(reference_path)

    # Evidently 0.7 uses Dataset objects plus a DataDefinition to understand
    # column roles during report execution.
    reference_data = Dataset.from_pandas(
        data=reference_df,
        data_definition=data_definition,
    )

    # window_size controls how many live rows are compared with the reference
    # data each time the report runs.
    window_size = config["service"].get("window_size", 250)

    # Exporting a reference hash helps confirm which baseline dataset is loaded.
    ref_hash = compute_hash(reference_df)
    logging.info(f"Reference data hash: {ref_hash}")

    # Export to Prometheus gauge.
    ref_hash_metric = prometheus_client.Gauge(
        "evidently_reference_dataset_hash",
        "Hash of the reference dataset used for monitoring",
        labelnames=["hash"],
    )
    ref_hash_metric.labels(hash=ref_hash).set(1)

    # Create the reusable report definition once at startup.
    report = create_report(config)

    # app.config is Flask's shared application dictionary. It keeps these objects
    # available to request handlers without reloading config or reference data.
    app.config.update(
        {
            "EVIDENTLY_CONFIG": config,
            "DATA_DEFINITION": data_definition,
            "REFERENCE_DATA": reference_data,
            "REPORT": report,
            "WINDOW_SIZE": window_size,
        },
    )

    logging.info("Evidently initialized successfully.")
    logging.info(f"Reference dataset rows: {len(reference_df)}")


# Run initialization immediately at startup.
init_evidently()

# This service stores the live monitoring window in memory.
# The Dockerfile uses one Gunicorn worker so every request updates this same
# DataFrame. A production service should use a shared durable store instead.
current_data = pd.DataFrame()

# Gauges are cached here so each Prometheus metric is registered only once.
gauge_store: dict[str, Any] = {}


@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset: str) -> str:
    global current_data
    global gauge_store

    # Step 1: Get one prediction event from the API.
    new_row = request.json
    if not isinstance(new_row, dict):
        return "Invalid payload. Expected a JSON object.", 400
    incoming_df = pd.DataFrame([new_row])

    # Step 2: Append the event to the current live window.
    current_data = pd.concat([current_data, incoming_df], ignore_index=True)

    window_size = app.config["WINDOW_SIZE"]
    # Keep only the latest N rows so the report represents recent traffic.
    if len(current_data) > window_size:
        current_data = current_data.tail(window_size).reset_index(drop=True)

    # Log progress.
    logging.info(f"Buffered {len(current_data)}/{window_size} rows")

    if len(current_data) < window_size:
        remaining = window_size - len(current_data)
        logging.info(f"Waiting for {remaining} more rows before running analysis.")
        return f"Buffering data... {len(current_data)}/{window_size}", 200

    # Once the window is full, compare current traffic with the reference data
    # and refresh the exported Prometheus metrics.
    logging.info("Window filled - starting drift analysis.")
    run_evidently(
        reference_data=app.config["REFERENCE_DATA"],
        current_data=current_data,
        data_definition=app.config["DATA_DEFINITION"],
        report=app.config["REPORT"],
        gauge_store=gauge_store,
        dataset_name=dataset,
    )

    return "ok"


@app.route("/report", methods=["GET"])
def view_report() -> ResponseReturnValue:
    # The report does not exist until the first full live window has been
    # analyzed. Returning 404 keeps that startup state explicit.
    if not Path("latest_report.html").exists():
        return "Report not generated yet", 404
    return send_file("latest_report.html")


if __name__ == "__main__":
    # debug=True is useful for development but should be disabled in production
    app.run(debug=True)  # noqa: S201
