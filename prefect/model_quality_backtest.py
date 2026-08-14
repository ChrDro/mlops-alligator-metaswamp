"""
Scheduled backtest of the key models' classification quality.

Why this is a scheduled flow rather than part of the serving path:
F1/precision/recall/log-loss score predictions against ground truth, and ground
truth does not exist at prediction time. Nobody hand-labels the live
new_predict_data columns, so these metrics cannot be computed from real traffic.
The only labelled data available is the frozen training set.

So this flow replays the *holdout* half of that labelled set - rows deliberately
excluded from each baseline by build_monitoring_references.py - through the live
models and posts the scored events to the Evidently service. The resulting
type="current" series answers a narrower but honest question: "do the model
versions serving right now still score the way their baselines did?"

That catches model-version regressions, a broken feature pipeline, and a bad
rollback. It does NOT catch real-world drift, because the input rows never change.
The drift dashboard covers that, fed automatically from the predict endpoints.

These numbers are NOT generalization quality - read them as a canary
--------------------------------------------------------------------
"Excluded from each baseline" means excluded from the Evidently *reference*, not
from *training*. The two splits are unrelated: build_monitoring_references.py
splits by hashing a column's identity, while the training scripts split by
`database`. Worse, the key models are refit on every labelled row before being
registered (see task_2/task_2_fk_train_and_register.py), so as of 2026-08-05
**100% of these holdout rows were in the training set of the model scoring them.**

The absolute values are therefore an in-sample upper bound. For fk_model the gap is
large and worth remembering: this backtest reports ~0.99 accuracy where the honest
out-of-fold estimate is F1 0.81.

That does not make the flow useless - a canary only has to be *consistent*, and
scoring the same rows every run is exactly what makes a drop mean "something
broke". It does mean the level carries no information about unseen data. The honest
quality number is the cross-validated one the training script logs to MLflow
(`cv_f1_mean`, with `cv_f1_std` for its spread), which the model service republishes
as the `model_offline_f1` gauge so Grafana can show both side by side.

Closing the gap properly would mean holding a grouped fold back from the final
refit. Measured cost for fk_model: -0.008 F1 on the served model, in exchange for
an estimate with *higher* variance than the 5-fold CV already gives - so the
trade was declined deliberately rather than overlooked.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from prefect import flow, task


MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model-service:8080")
MONITORING_BASE_URL = os.getenv("MONITORING_BASE_URL", "http://evidently_service:8085")

# ./data is mounted read-only into the prefect container by docker-compose.yaml.
HOLDOUT_DIR = Path(os.getenv("HOLDOUT_DIR", "/opt/data/holdouts"))

# One entry per monitored model, matching MODEL_SPECS in
# evidently_service/build_monitoring_references.py, which writes the holdout files
# this reads. Only the endpoint and target are named here - the feature list comes
# from the holdout file's own columns, so it cannot drift out of sync.
TRACKS: dict[str, dict[str, str]] = {
    "pk_columns": {"endpoint": "/predict_pk", "target": "pk_target"},
    "cpk_columns": {"endpoint": "/predict_cpk", "target": "composite_pk_target"},
    "fk_columns": {"endpoint": "/predict_fk", "target": "fk_target"},
    "cfk_columns": {"endpoint": "/predict_cfk", "target": "composite_fk_target"},
    # Multiclass: normal forms 0-3. Scored on labels alone, so it needs no special
    # handling here - the Evidently service decides which metrics apply.
    "nf_columns": {"endpoint": "/predict_normalform", "target": "target_normal_form"},
}

# Written into the holdout files for traceability; not model features.
IDENTITY_COLUMNS = {"database", "schema", "table_name", "column_name", "column_type"}


@task
def load_holdout(track: str, target: str, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Load a stratified sample of one track's labelled holdout rows.

    Stratified because these targets are rare - composite_fk_target is only ~2%
    positive - so an unstratified sample of a few dozen rows can contain no
    positives at all, and the Evidently service skips single-class windows rather
    than exporting a misleading zero. It matters for nf_columns too, where macro
    averaging is only meaningful if every normal form appears in the window.
    """
    path = HOLDOUT_DIR / f"{track}.csv"
    if not path.exists():
        msg = (
            f"Holdout file not found at {path}. Generate it with "
            f"evidently_service/build_monitoring_references.py, and check that ./data "
            f"is mounted into this container."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)

    if sample_size and sample_size < len(df):
        fraction = sample_size / len(df)
        strata = [
            group.sample(n=max(1, round(len(group) * fraction)), random_state=seed)
            for _, group in df.groupby(target)
        ]
        df = pd.concat(strata).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Class counts rather than a positive count: nf_columns has four classes, so
    # "positive" is meaningless there.
    counts = df[target].value_counts().sort_index().to_dict()
    print(f"[{track}] {len(df)} holdout rows, class counts={counts}")
    return df


@task
def replay_track(track: str, spec: dict[str, str], holdout: pd.DataFrame) -> dict[str, Any]:
    """
    Score each row through the live model and post it with its label.

    Individual failures are counted rather than raised: a few bad rows should not
    void the whole backtest, but a total failure needs to surface as one.
    """
    target = spec["target"]
    features = [c for c in holdout.columns if c != target and c not in IDENTITY_COLUMNS]

    session = requests.Session()
    predict_url = f"{MODEL_API_URL}{spec['endpoint']}"
    monitor_url = f"{MONITORING_BASE_URL}/iterate_classification/{track}"

    sent = 0
    failures = 0
    scored_windows = 0
    last_metrics: dict[str, Any] = {}
    targets = holdout[target].astype(int).tolist()

    for position, (_, row) in enumerate(holdout[features].iterrows()):
        payload = {
            key: bool(row[key])
            if key.startswith("column_type_")
            else (row[key].item() if hasattr(row[key], "item") else row[key])
            for key in features
        }

        try:
            prediction = session.post(predict_url, json=payload, timeout=30).json()

            event = dict(prediction)
            event[target] = targets[position]

            response = session.post(monitor_url, json=event, timeout=60)
            response.raise_for_status()
            sent += 1

            # A scored window returns JSON carrying metrics; a deferred one returns
            # a plain-text buffering message.
            if response.headers.get("content-type", "").startswith("application/json"):
                body = response.json()
                if isinstance(body, dict) and body.get("metrics"):
                    scored_windows += 1
                    last_metrics = body["metrics"]
        except requests.RequestException as error:
            failures += 1
            if failures <= 3:
                print(f"[{track}] row {position} failed: {error}")

    if sent == 0:
        msg = (
            f"[{track}] every one of {len(holdout)} rows failed. Check that "
            f"model-service and evidently_service are reachable from this container."
        )
        raise RuntimeError(msg)

    return {
        "track": track,
        "rows_sent": sent,
        "failures": failures,
        "windows_scored": scored_windows,
        "metrics": last_metrics,
    }


@flow(name="model-quality-backtest")
def model_quality_backtest(
    sample_size: int = 200,
    seed: int = 0,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """
    Replay labelled holdout rows through the live key models and score the results.

    Args:
        sample_size: Rows to replay per track. Should comfortably exceed that
            track's classification.window_size so at least one window closes.
            cfk_columns uses 150 because its target is so rare; nf_columns uses 100
            so every normal form lands in the window.
        seed: Sampling seed. Vary it between runs to cover different holdout rows;
            keep it fixed to compare like with like across model versions.
        only: Restrict to these tracks. Defaults to all five.

    Returns:
        Per-track counts plus the metrics from the last window that closed.
    """
    selected = only or list(TRACKS)
    unknown = [t for t in selected if t not in TRACKS]
    if unknown:
        msg = f"Unknown track(s) {unknown}. Known: {sorted(TRACKS)}"
        raise ValueError(msg)

    results = {}
    for track in selected:
        spec = TRACKS[track]
        holdout = load_holdout(track, spec["target"], sample_size, seed)
        results[track] = replay_track(track, spec, holdout)

    print("\nBacktest summary:")
    for track, result in results.items():
        if result["metrics"]:
            scores = "  ".join(f"{k}={v:.4f}" for k, v in sorted(result["metrics"].items()))
            print(f"  {track:<12} {result['windows_scored']} window(s)  {scores}")
        else:
            # Most likely every window was single-class, which is expected for the
            # rarest targets at small sample sizes.
            print(
                f"  {track:<12} no window closed "
                f"({result['rows_sent']} rows sent, {result['failures']} failures)",
            )

    return results


if __name__ == "__main__":
    model_quality_backtest()
