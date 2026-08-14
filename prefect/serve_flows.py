"""
Register and serve the streaming deployments.

Run this as a long-lived process. In docker-compose the ``prefect`` service starts
the server in the background and then runs this file in the foreground; locally,
``python prefect/serve_flows.py`` against a running server does the same.

``serve()`` registers the deployments with the Prefect server and then executes
triggered runs in this same process - there is no work pool and no Prefect worker
involved, despite what the name of that concept might suggest.

Event wiring
------------
The triggers are attached to the deployments themselves, so they are created and
kept in sync automatically whenever this process starts - there is no separate
"create the automations" step that can be forgotten::

    POST /events/new-data ─┐
                           ├─→ alligator.new-data.arrived ─→ change-detection-check
    cron (safety net) ─────┘                                          │
                                            diff watermarks → pending_changes
                                                              │
                                              alligator.changes.recorded
                                                              │
                        ┌───────────────────────────┼───────────────────────────┐
              key-prediction-        normalform-prediction-      subject-area-prediction-
                  pipeline                  pipeline                    pipeline
              (track: keys)              (track: nf)             (track: subject_area)

All three prediction deployments listen to the *same* event, so they run in parallel
and none can starve the others.

A fifth deployment, model-quality-backtest, is unrelated to this event chain.
It runs purely on a cron schedule and exists because F1/precision/recall need
ground-truth labels, which no live prediction carries - see
model_quality_backtest.py for why that track cannot be event-driven. It covers all
four key models in one run.
"""

from datetime import timedelta

from change_detector import change_detection_poller
from change_events import CHANGES_RECORDED_EVENT, EVENT_RESOURCE_ID, NEW_DATA_EVENT
from model_quality_backtest import model_quality_backtest
from normalform_pipeline import normalform_prediction_pipeline
from pk_fk_pipeline import feature_engineering_pipeline
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
from prefect.events import DeploymentEventTrigger
from subject_area_pipeline import subject_area_prediction_pipeline

from prefect import serve


TARGET_SCHEMAS = ["new_predict_data"]

# Safety net for everything the push path misses. Every 15 minutes is cheap: the
# detector only counts rows, it does not profile columns.
DETECTION_CRON = "*/15 * * * *"

# Hourly is plenty for the model-quality backtest. It replays a fixed labelled
# sample, so its result only moves when the served model version changes - running
# it more often would just re-measure the same thing.
BACKTEST_CRON = "17 * * * *"

# Only match our own events, so an unrelated event can never start a pipeline.
RESOURCE_MATCH = {"prefect.resource.id": EVENT_RESOURCE_ID}


detector_deployment = change_detection_poller.to_deployment(
    name="streaming",
    cron=DETECTION_CRON,
    parameters={"target_schemas": TARGET_SCHEMAS, "source": "poller"},
    # CANCEL_NEW debounces bursts: while a detection run is active, further pushes
    # are dropped rather than queued, because the running snapshot already sees the
    # current state of every table. A change landing mid-snapshot is caught by the
    # next cron pass - that is exactly what the schedule is there for.
    concurrency_limit=ConcurrencyLimitConfig(
        limit=1,
        collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
    ),
    triggers=[
        DeploymentEventTrigger(
            name="on-new-data-pushed",
            expect={NEW_DATA_EVENT},
            match=RESOURCE_MATCH,
            within=timedelta(seconds=0),
            parameters={"target_schemas": TARGET_SCHEMAS, "source": "webhook"},
        ),
    ],
    tags=["streaming", "detector"],
    description="Diff table watermarks and record what needs re-predicting.",
)


keys_deployment = feature_engineering_pipeline.to_deployment(
    name="streaming",
    parameters={"target_schemas": TARGET_SCHEMAS, "use_pending_changes": True},
    # ENQUEUE, not CANCEL_NEW: a dropped run here could strand pending changes that
    # no later event would reference again. A queued duplicate is cheap - it claims
    # an empty work list and exits in seconds.
    concurrency_limit=ConcurrencyLimitConfig(
        limit=1,
        collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
    ),
    triggers=[
        DeploymentEventTrigger(
            name="on-changes-recorded",
            expect={CHANGES_RECORDED_EVENT},
            match=RESOURCE_MATCH,
            within=timedelta(seconds=0),
        ),
    ],
    tags=["streaming", "keys"],
    description="Re-profile and re-predict pk/fk/cpk/cfk for changed tables.",
)


normalform_deployment = normalform_prediction_pipeline.to_deployment(
    name="streaming",
    parameters={"target_schemas": TARGET_SCHEMAS, "use_pending_changes": True},
    concurrency_limit=ConcurrencyLimitConfig(
        limit=1,
        collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
    ),
    triggers=[
        DeploymentEventTrigger(
            name="on-changes-recorded",
            expect={CHANGES_RECORDED_EVENT},
            match=RESOURCE_MATCH,
            within=timedelta(seconds=0),
        ),
    ],
    tags=["streaming", "normalform"],
    description="Re-profile and re-predict the normal form of changed tables.",
)


subject_area_deployment = subject_area_prediction_pipeline.to_deployment(
    name="streaming",
    parameters={"target_schemas": TARGET_SCHEMAS, "use_pending_changes": True},
    concurrency_limit=ConcurrencyLimitConfig(
        limit=1,
        collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
    ),
    triggers=[
        DeploymentEventTrigger(
            name="on-changes-recorded",
            expect={CHANGES_RECORDED_EVENT},
            match=RESOURCE_MATCH,
            within=timedelta(seconds=0),
        ),
    ],
    tags=["streaming", "subject-area"],
    description="Re-predict the subject area of changed tables.",
)


backtest_deployment = model_quality_backtest.to_deployment(
    name="scheduled",
    cron=BACKTEST_CRON,
    parameters={"sample_size": 200},
    # CANCEL_NEW: overlapping runs would interleave their rows in the Evidently
    # service's single in-memory window, mixing two samples into one score.
    concurrency_limit=ConcurrencyLimitConfig(
        limit=1,
        collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
    ),
    tags=["monitoring", "model-quality"],
    description="Replay labelled holdout rows through the live key models and score F1.",
)


if __name__ == "__main__":
    serve(
        detector_deployment,
        keys_deployment,
        normalform_deployment,
        subject_area_deployment,
        backtest_deployment,
        limit=5,
    )
