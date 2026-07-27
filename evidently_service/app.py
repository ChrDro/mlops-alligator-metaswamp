import logging
from pathlib import Path
from typing import Any

import pandas as pd
import prometheus_client
from flask import Flask, request, send_file
from flask.typing import ResponseReturnValue
from utils import (
    Dataset,
    build_classification_data_definition,
    build_data_definition,
    compute_hash,
    create_classification_report,
    create_report,
    load_config,
    positive_class_probability,
    run_classification_report,
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

    init_classification_track(config, base_dir)


def init_classification_track(config: dict[str, Any], base_dir: Path) -> None:
    """
    Set up the classification-quality track and export the reference baseline.

    The reference scores are static, so they are computed once here rather than on
    every request. A missing reference file is not fatal: the live window still
    reports, just without a baseline to compare against.
    """
    clf_config = config.get("classification")
    if not clf_config or not clf_config.get("enabled", False):
        logging.info("Classification track disabled - skipping.")
        app.config["CLASSIFICATION_ENABLED"] = False
        return

    clf_definition = build_classification_data_definition(config)
    clf_report = create_classification_report(config)

    app.config.update(
        {
            "CLASSIFICATION_ENABLED": True,
            "CLASSIFICATION_CONFIG": clf_config,
            "CLASSIFICATION_DEFINITION": clf_definition,
            "CLASSIFICATION_REPORT": clf_report,
            "CLASSIFICATION_WINDOW_SIZE": clf_config.get("window_size", 50),
        },
    )

    reference_path = Path(clf_config["reference_path"])
    if not reference_path.is_absolute():
        reference_path = base_dir / reference_path

    if not reference_path.exists():
        logging.warning(
            f"Classification reference not found at {reference_path}. Current-window "
            f'metrics will still be exported, but no type="reference" series will '
            f"exist. Generate it with build_classification_reference.py.",
        )
        return

    reference_df = pd.read_csv(reference_path)
    required = {clf_config["target"], clf_config["prediction"], clf_config["prediction_proba"]}
    missing = required - set(reference_df.columns)
    if missing:
        logging.error(
            f"Classification reference {reference_path} is missing columns {sorted(missing)}. "
            f"Skipping the reference series.",
        )
        return

    # save_html=False: the reference pass would otherwise write a report file that
    # the first live window immediately overwrites.
    run_classification_report(
        scored_data=reference_df,
        data_definition=clf_definition,
        report=clf_report,
        gauge_store=clf_gauge_store,
        dataset_name="reference",
        series_type="reference",
        save_html=False,
    )
    logging.info(f"Classification reference scored: {len(reference_df)} rows.")


# These live-window buffers and gauge caches are declared before init_evidently()
# runs, because scoring the classification reference at startup already writes
# into clf_gauge_store.
#
# This service stores the live monitoring window in memory. The Dockerfile uses
# one Gunicorn worker so every request updates this same DataFrame. A production
# service should use a shared durable store instead.
current_data = pd.DataFrame()

# Separate buffer: classification events carry ground truth, drift events do not,
# so the two windows fill independently and must not be mixed.
current_classification_data = pd.DataFrame()

# Gauges are cached here so each Prometheus metric is registered only once.
gauge_store: dict[str, Any] = {}
clf_gauge_store: dict[str, Any] = {}

# Run initialization immediately at startup.
init_evidently()


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


def prepare_classification_event(
    payload: Any,
    target_col: str,
    prediction_col: str,
    proba_col: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validate one incoming labelled event and fill in P(class=1).

    Returns ``(row, None)`` when the event is usable, or ``(None, reason)`` when it
    is not. Collecting the checks here keeps the request handler to a single
    rejection path.
    """
    if not isinstance(payload, dict):
        return None, "Invalid payload. Expected a JSON object."

    # Ground truth is what separates this track from /iterate. Without it the event
    # is unscoreable, so reject it loudly rather than buffering a row that would
    # later break the report.
    missing = [c for c in (target_col, prediction_col) if c not in payload]
    if missing:
        return None, f"Payload is missing required field(s): {missing}"

    row = dict(payload)

    # Accept a caller-supplied P(class=1) if present; otherwise derive it from the
    # model API's confidence value.
    if proba_col not in row:
        if "probability" not in row:
            return None, f"Payload needs either {proba_col!r} or 'probability'"
        row[proba_col] = positive_class_probability(row[prediction_col], row["probability"])

    return row, None


@app.route("/iterate_classification/<dataset>", methods=["POST"])
def iterate_classification(dataset: str) -> ResponseReturnValue:
    """
    Buffer one *labelled* prediction event and score the window when it fills.

    Expected JSON body: the POST /predict_pk response object (which echoes the
    features and appends ``prediction`` and ``probability``) plus the ground-truth
    label under the configured target key, e.g.::

        {..., "prediction": 1, "probability": 0.93, "pk_target": 1}

    ``probability`` is the confidence in the predicted class, so this endpoint
    converts it to P(class=1) before storing - see positive_class_probability.
    """
    global current_classification_data

    if not app.config.get("CLASSIFICATION_ENABLED", False):
        return "Classification track is disabled in config.yaml", 503

    clf_config = app.config["CLASSIFICATION_CONFIG"]
    target_col = clf_config["target"]
    prediction_col = clf_config["prediction"]
    proba_col = clf_config["prediction_proba"]

    row, rejection = prepare_classification_event(
        request.json,
        target_col,
        prediction_col,
        proba_col,
    )
    if rejection is not None:
        return rejection, 400

    current_classification_data = pd.concat(
        [current_classification_data, pd.DataFrame([row])],
        ignore_index=True,
    )

    window_size = app.config["CLASSIFICATION_WINDOW_SIZE"]
    if len(current_classification_data) > window_size:
        current_classification_data = current_classification_data.tail(window_size).reset_index(
            drop=True,
        )

    buffered = len(current_classification_data)
    logging.info(f"Buffered {buffered}/{window_size} labelled rows")

    if buffered < window_size:
        return f"Buffering labelled data... {buffered}/{window_size}", 200

    # Log loss is undefined when the window holds a single class, and precision or
    # recall would be degenerate. Keep buffering rather than exporting a
    # misleading zero.
    if current_classification_data[target_col].nunique() < 2:
        logging.warning(
            f"Window of {buffered} rows contains only one class in {target_col!r} - "
            f"skipping this run. Log loss and recall need both classes present.",
        )
        return f"Window is single-class in {target_col} - waiting for both classes", 200

    logging.info("Classification window filled - scoring against ground truth.")
    metrics = run_classification_report(
        scored_data=current_classification_data,
        data_definition=app.config["CLASSIFICATION_DEFINITION"],
        report=app.config["CLASSIFICATION_REPORT"],
        gauge_store=clf_gauge_store,
        dataset_name=dataset,
        series_type="current",
    )
    return {"status": "ok", "rows_scored": buffered, "metrics": metrics}


@app.route("/classification_report", methods=["GET"])
def view_classification_report() -> ResponseReturnValue:
    if not Path("latest_classification_report.html").exists():
        return "Classification report not generated yet", 404
    return send_file("latest_classification_report.html")


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
