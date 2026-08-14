"""Evidently monitoring service.

Serves two independent monitoring tracks per model, both exposed on /metrics for
Prometheus:

  drift            POST /iterate/<track>
                   Compares live feature and prediction distributions against a
                   frozen reference. Needs no labels, so the prediction endpoints
                   forward every request here as a background task.

  classification   POST /iterate_classification/<track>
                   Scores predictions against ground truth (F1/precision/recall/
                   accuracy/log loss). Needs labels, which live predictions do not
                   carry, so this is driven by the Prefect backtest instead.

Each configured model gets its own DataDefinition, reference dataset, report and
rolling window - held in TRACKS below. The track name is both the path segment and
the dataset_name label on every gauge, which is what lets one Grafana dashboard
switch between models with a variable.
"""

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
    infer_column_roles,
    load_config,
    positive_class_probability,
    run_classification_report,
    run_evidently,
)
from werkzeug.middleware.dispatcher import DispatcherMiddleware


# ------------------------------------------------------------------------------
# Flask + Prometheus setup
# ------------------------------------------------------------------------------

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

prometheus_metrics_app = prometheus_client.make_wsgi_app()

# Any request to /metrics is routed to the Prometheus WSGI app instead of Flask.
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": prometheus_metrics_app})


# One entry per configured model, keyed by track name. Holds the mutable live
# windows, so it lives at module level rather than in app.config.
TRACKS: dict[str, dict[str, Any]] = {}

# Gauge caches are GLOBAL, not per track. Every track produces the same metric
# names (evidently_valuedrift_prediction and so on) and they are separated by the
# dataset_name label, so one Gauge object must be shared - registering the same
# metric name twice is an error in prometheus_client.
gauge_store: dict[str, Any] = {}
clf_gauge_store: dict[str, Any] = {}


def load_track(track: str, track_config: dict[str, Any], base_dir: Path) -> dict[str, Any] | None:
    """
    Build one track's monitoring state from its reference file.

    Returns None if the track cannot be monitored, which is not fatal for the
    others: a missing reference is the normal state before
    build_monitoring_references.py has been run.
    """
    reference_path = Path(track_config["reference_path"])
    if not reference_path.is_absolute():
        reference_path = base_dir / reference_path

    if not reference_path.exists():
        logging.warning(
            f"[{track}] reference not found at {reference_path} - track disabled. "
            f"Generate it with build_monitoring_references.py.",
        )
        return None

    reference_df = pd.read_csv(reference_path)
    target = track_config["target"]
    prediction = track_config["prediction"]

    if target not in reference_df.columns or prediction not in reference_df.columns:
        logging.error(
            f"[{track}] reference is missing {target!r} or {prediction!r} - track disabled.",
        )
        return None

    numerical, categorical = infer_column_roles(
        reference_df,
        target=target,
        force_categorical=track_config.get("force_categorical", ()),
    )
    definition = build_data_definition(
        numerical_columns=numerical,
        categorical_columns=categorical,
        prediction=prediction,
        prediction_type=track_config.get("prediction_type", "numerical"),
    )

    drift_columns = list(track_config.get("drift_columns", [prediction]))
    report = create_report(
        drift_columns=drift_columns,
        prediction=prediction,
        known_columns=[*numerical, *categorical],
    )

    state: dict[str, Any] = {
        "name": track,
        "target": target,
        "prediction": prediction,
        "definition": definition,
        "report": report,
        "reference": Dataset.from_pandas(reference_df, data_definition=definition),
        # Kept alongside the Dataset for the hash gauge and /tracks; avoids
        # depending on an accessor to read the frame back out of Evidently.
        "reference_df": reference_df,
        "window_size": track_config.get("window_size", 60),
        "report_every": track_config.get("report_every", track_config.get("window_size", 60)),
        "buffer": pd.DataFrame(),
        "rows_since_report": 0,
        # Populated below when classification is enabled.
        "clf": None,
    }

    logging.info(
        f"[{track}] {len(reference_df)} reference rows | "
        f"{len(numerical)} numerical + {len(categorical)} categorical features | "
        f"drift columns: {drift_columns}",
    )

    clf_config = track_config.get("classification") or {}
    if clf_config.get("enabled", False):
        state["clf"] = load_classification_track(track, track_config, clf_config, reference_df)

    return state


def load_classification_track(
    track: str,
    track_config: dict[str, Any],
    clf_config: dict[str, Any],
    reference_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """
    Set up one track's classification state and export its reference baseline.

    The reference scores are static, so they are computed once here rather than on
    every request. Evidently's snapshot only ever reports the current dataset -
    passing reference_data does not add reference values to it - so the two series
    are produced by scoring the two datasets separately.
    """
    target = track_config["target"]
    prediction = track_config["prediction"]
    multiclass = clf_config.get("multiclass", False)

    # Multiclass tracks have no probability column by design: the API returns
    # max(predict_proba), which for >2 classes cannot be expanded back into a full
    # distribution. Log loss is therefore not offered for them.
    proba = None if multiclass else clf_config["prediction_proba"]

    if proba is not None and proba not in reference_df.columns:
        logging.error(
            f"[{track}] reference is missing {proba!r} - classification track disabled. "
            f"Rebuild the reference with build_monitoring_references.py.",
        )
        return None

    definition = build_classification_data_definition(
        target,
        prediction,
        prediction_proba=proba,
        multiclass=multiclass,
    )
    report = create_classification_report(multiclass=multiclass)

    window_size = clf_config.get("window_size", 50)
    state = {
        "definition": definition,
        "report": report,
        "proba": proba,
        "multiclass": multiclass,
        "window_size": window_size,
        "report_every": clf_config.get("report_every", window_size),
        "buffer": pd.DataFrame(),
        "rows_since_report": 0,
    }

    if reference_df[target].nunique(dropna=True) < 2:
        logging.error(
            f"[{track}] reference contains only one class in {target!r} - cannot score a "
            f"baseline. Rebuild it with a larger --limit so both classes appear.",
        )
        return state

    # save_html=False: the reference pass would otherwise write a report file that
    # the first live window immediately overwrites.
    run_classification_report(
        scored_data=reference_df,
        data_definition=definition,
        report=report,
        gauge_store=clf_gauge_store,
        dataset_name=track,
        series_type="reference",
        save_html=False,
    )
    logging.info(f"[{track}] classification reference scored ({len(reference_df)} rows)")
    return state


def init_evidently() -> None:
    """Load config and build every configured track."""
    logging.info("Initializing Evidently monitoring service...")

    # Build paths from this file's location so the service works whether it is
    # launched from the repo root or from inside the Docker container.
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / "config.yaml")

    models = config.get("models") or {}
    if not models:
        logging.error("config.yaml declares no models - nothing to monitor.")
        return

    for track, track_config in models.items():
        state = load_track(track, track_config, base_dir)
        if state is not None:
            TRACKS[track] = state

    if not TRACKS:
        logging.warning(
            "No track could be initialised. /iterate will return 404 for every track "
            "until at least one reference file exists.",
        )
        return

    # A hash of each reference makes it possible to confirm from Prometheus alone
    # which baseline a running service actually loaded.
    hash_metric = prometheus_client.Gauge(
        "evidently_reference_dataset_hash",
        "Hash of the reference dataset used for monitoring",
        labelnames=["dataset_name", "hash"],
    )
    for track, state in TRACKS.items():
        ref_hash = compute_hash(state["reference_df"])
        hash_metric.labels(dataset_name=track, hash=ref_hash).set(1)

    logging.info(f"Evidently initialized. Active tracks: {sorted(TRACKS)}")


init_evidently()


def get_track(track: str) -> dict[str, Any] | None:
    return TRACKS.get(track)


def unknown_track_response(track: str) -> tuple[str, int]:
    message = (
        f"Unknown track {track!r}. Configured and active: {sorted(TRACKS)}. "
        f"A track is inactive if its reference file is missing."
    )
    return message, 404


@app.route("/iterate/<track>", methods=["POST"])
def iterate(track: str) -> ResponseReturnValue:
    """
    Buffer one prediction event and run the drift report when due.

    Expected body: the prediction endpoint's response object, which echoes the
    request features and appends ``prediction`` and ``probability``. No labels
    needed - that is what lets the serving path call this directly.
    """
    state = get_track(track)
    if state is None:
        return unknown_track_response(track)

    new_row = request.json
    if not isinstance(new_row, dict):
        return "Invalid payload. Expected a JSON object.", 400

    state["buffer"] = pd.concat(
        [state["buffer"], pd.DataFrame([new_row])],
        ignore_index=True,
    )
    state["rows_since_report"] += 1

    window_size = state["window_size"]
    # Keep only the latest N rows so the report represents recent traffic.
    if len(state["buffer"]) > window_size:
        state["buffer"] = state["buffer"].tail(window_size).reset_index(drop=True)

    buffered = len(state["buffer"])
    if buffered < window_size:
        return f"Buffering data... {buffered}/{window_size}", 200

    # The window is a rolling tail, so it stays full from here on. Without this gate
    # every later request would run a full report over every monitored column - one
    # per prediction, since the predict endpoints forward them all.
    report_every = state["report_every"]
    if state["rows_since_report"] < report_every:
        remaining = report_every - state["rows_since_report"]
        return f"Buffered; next report in {remaining} rows", 200
    state["rows_since_report"] = 0

    logging.info(f"[{track}] running drift analysis over the latest {buffered} rows")
    run_evidently(
        reference_data=state["reference"],
        current_data=state["buffer"],
        data_definition=state["definition"],
        report=state["report"],
        gauge_store=gauge_store,
        dataset_name=track,
        html_path=f"latest_report_{track}.html",
    )
    return "ok"


def prepare_classification_event(
    payload: Any,
    target_col: str,
    prediction_col: str,
    proba_col: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validate one incoming labelled event and fill in P(class=1) where applicable.

    Returns ``(row, None)`` when the event is usable, or ``(None, reason)`` when it
    is not. Collecting the checks here keeps the request handler to a single
    rejection path.

    Args:
        proba_col: None for multiclass tracks, which score from hard labels only.
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

    # Binary tracks only. Accept a caller-supplied P(class=1) if present; otherwise
    # derive it from the model API's confidence value.
    if proba_col is not None and proba_col not in row:
        if "probability" not in row:
            return None, f"Payload needs either {proba_col!r} or 'probability'"
        row[proba_col] = positive_class_probability(row[prediction_col], row["probability"])

    return row, None


def classification_defer_reason(
    clf: dict[str, Any],
    track: str,
    target_col: str,
) -> str | None:
    """
    Decide whether to score the classification window now.

    Returns a human-readable reason to defer, or None to go ahead. Collecting the
    three deferral cases here keeps the request handler to one deferral path.
    """
    buffered = len(clf["buffer"])
    window_size = clf["window_size"]
    if buffered < window_size:
        return f"Buffering labelled data... {buffered}/{window_size}"

    report_every = clf["report_every"]
    if clf["rows_since_report"] < report_every:
        return f"Buffered; next scoring in {report_every - clf['rows_since_report']} rows"

    # Log loss is undefined when the window holds a single class, and precision or
    # recall would be degenerate. Defer rather than export a misleading zero. The
    # counter is deliberately left untouched so the next row retries immediately
    # instead of waiting another full report_every rows. Rare targets like
    # composite_fk_target (~2% positive) hit this often - widen that track's
    # classification.window_size if it never clears.
    if clf["buffer"][target_col].nunique() < 2:
        logging.warning(
            f"[{track}] window of {buffered} rows is single-class in {target_col!r} - "
            f"skipping. Log loss and recall need both classes present.",
        )
        return f"Window is single-class in {target_col} - waiting for both classes"

    return None


@app.route("/iterate_classification/<track>", methods=["POST"])
def iterate_classification(track: str) -> ResponseReturnValue:
    """
    Buffer one *labelled* prediction event and score the window when due.

    Expected body: the prediction endpoint's response object plus the ground-truth
    label under the track's configured target key, e.g.::

        {..., "prediction": 1, "probability": 0.93, "pk_target": 1}

    ``probability`` is the confidence in the predicted class, so this endpoint
    converts it to P(class=1) before storing - see positive_class_probability.
    """
    state = get_track(track)
    if state is None:
        return unknown_track_response(track)

    clf = state["clf"]
    if clf is None:
        return f"Classification track is not enabled for {track!r}", 503

    target_col = state["target"]
    prediction_col = state["prediction"]

    row, rejection = prepare_classification_event(
        request.json,
        target_col,
        prediction_col,
        clf["proba"],
    )
    if rejection is not None:
        return rejection, 400

    clf["buffer"] = pd.concat([clf["buffer"], pd.DataFrame([row])], ignore_index=True)
    clf["rows_since_report"] += 1

    window_size = clf["window_size"]
    if len(clf["buffer"]) > window_size:
        clf["buffer"] = clf["buffer"].tail(window_size).reset_index(drop=True)

    buffered = len(clf["buffer"])
    defer = classification_defer_reason(clf, track, target_col)
    if defer is not None:
        return defer, 200

    clf["rows_since_report"] = 0
    logging.info(f"[{track}] scoring the latest {buffered} labelled rows against ground truth")
    metrics = run_classification_report(
        scored_data=clf["buffer"],
        data_definition=clf["definition"],
        report=clf["report"],
        gauge_store=clf_gauge_store,
        dataset_name=track,
        series_type="current",
        html_path=f"latest_classification_report_{track}.html",
    )
    return {"status": "ok", "track": track, "rows_scored": buffered, "metrics": metrics}


@app.route("/tracks", methods=["GET"])
def list_tracks() -> ResponseReturnValue:
    """Report what is being monitored, for debugging a dashboard with no data."""
    return {
        "tracks": {
            name: {
                "target": state["target"],
                "reference_rows": len(state["reference_df"]),
                "drift_window": state["window_size"],
                "drift_report_every": state["report_every"],
                "drift_buffered": len(state["buffer"]),
                "classification_enabled": state["clf"] is not None,
                "classification_buffered": len(state["clf"]["buffer"]) if state["clf"] else 0,
            }
            for name, state in TRACKS.items()
        },
    }


@app.route("/report/<track>", methods=["GET"])
def view_report(track: str) -> ResponseReturnValue:
    # The report does not exist until the first full window has been analyzed.
    # Returning 404 keeps that startup state explicit.
    path = Path(f"latest_report_{track}.html")
    if not path.exists():
        return f"Drift report for {track!r} not generated yet", 404
    return send_file(path)


@app.route("/classification_report/<track>", methods=["GET"])
def view_classification_report(track: str) -> ResponseReturnValue:
    path = Path(f"latest_classification_report_{track}.html")
    if not path.exists():
        return f"Classification report for {track!r} not generated yet", 404
    return send_file(path)


if __name__ == "__main__":
    # debug=True is useful for development but should be disabled in production
    app.run(debug=True)  # noqa: S201
