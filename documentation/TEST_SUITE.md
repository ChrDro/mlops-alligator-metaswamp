# Test Suite — Webservice & Pipelines

Catalogue of the 217 tests added on 2026-07-30 (plan items 2.2 / P5–P10). Companion to
[`test/test_data/README.md`](../test/test_data/README.md), which documents the older
training-data suite.

**Scope of this document:** the two new packages
[`test/test_webservice/`](../test/test_webservice/) and
[`test/test_pipelines/`](../test/test_pipelines/). The pre-existing
[`test/test_data/`](../test/test_data/) (data quality, 46 tests) and
[`test/test_models/`](../test/test_models/) (registry contract, 10 tests) are unchanged.

## Running them

```bash
uv sync                          # test tooling is a main dependency, no flag needed
pytest                           # all 273
pytest -m "not integration"      # 263, no docker stack needed — what the git hook runs
pytest --cov                     # coverage, gated at 80% via [tool.coverage.report]
```

Every test in this document runs **without a stack**: no MLflow, no Trino, no Evidently,
no Prefect server, no network. Only `test/test_models/` needs the compose stack, and it
is marked `integration` so it can be deselected.

| Package | Tests | Functions | Needs stack |
| :--- | ---: | ---: | :--- |
| `test/test_webservice/` | 147 | 56 | no |
| `test/test_pipelines/` | 70 | 47 | no |
| **Total (new)** | **217** | **103** | |

Test counts exceed function counts because many are parametrised — most often across all
five models, since the five predict handlers are copy-paste clones (open item 3.1) and a
slip in one of them is the likeliest defect in that file.

---

## Shared test infrastructure

### `test/test_webservice/conftest.py`

| Helper | Purpose |
| :--- | :--- |
| `sys.path` insert | The webservice is not an installable package — `app.py` imports its siblings as top-level modules, so `webservice/` goes on the path (same approach as `test_models/`). |
| `payload_for(model_cls)` | Builds a valid request body from `model_fields`. The five request models carry 32–52 features each; a hand-written fixture would be one more place to forget on a schema change — exactly the failure behind the 27.07. drift. |
| `FakePyFuncModel` | Stands in for an MLflow `PyFuncModel`. Keeps the **split** that caused the 27.07. bug: `predict()` (pyfunc, reorders by name) and `_model_impl.get_raw_model().predict_proba()` (raw sklearn, compares positionally) are separate objects that each record the column order they saw. |

### `test/test_pipelines/conftest.py`

| Helper | Purpose |
| :--- | :--- |
| `sys.path` insert | Adds `prefect/` — **not** the repo root, which contains a directory literally named `prefect` that would shadow the installed Prefect library and break `from prefect import flow, task` inside the very modules under test. |
| `FakeConnection` | Records every `execute()` as `(sql, params)` and returns canned rows. This is what makes it possible to assert that table names travel as **bound parameters** rather than interpolated SQL — something a live database could not show as clearly. |
| `FakeResult` | Minimal `CursorResult`: `fetchall()`, `scalar()`, `rowcount`. |

---

## `test_predict.py` — 12 tests

Prediction logic. Both halves of this module caused a production outage; both regressions
are pinned.

| Test | Asserts | Why |
| :--- | :--- | :--- |
| `test_align_reorders_columns_into_signature_order` | Columns come back in signature order, values travel with their column | The core of the fix — not just labels reordered |
| `test_align_passes_frame_through_when_model_has_no_signature` | No signature → frame untouched | A model logged without a signature must still be servable |
| `test_align_error_names_the_missing_features` | Error names the model, the missing field, and `Unexpected: none` | The bare sklearn message says nothing about *which* field drifted |
| `test_align_error_names_the_unexpected_features` | Error names the surplus field | Same, in the other direction |
| `test_predict_returns_probability_of_the_predicted_class` | 4-class model predicting class 3 returns **0.55** | **Regression test for the 28.07. bug.** Built so the old `probabilities[0][1]` (0.20) and the correct `max(proba)` (0.55) differ — a reintroduced fixed class index fails here |
| `test_predict_returns_probability_of_the_predicted_class_for_binary` | Predicting class 0 returns p(class 0) = 0.85, not p(class 1) | The old expression happened to look right for binary models predicting 1; this catches it |
| `test_predict_sends_identical_column_order_to_pyfunc_and_to_predict_proba` | Both objects see `["c","a","b"]` | **The 27.07. bug itself**: pyfunc reorders by name, the raw estimator does not |
| `test_predict_casts_booleans_to_numeric` | All dtypes arrive as `float64` | Request schemas declare `column_type_*` as `bool`; models were trained on 0/1 |
| `test_predict_does_not_mutate_the_caller_frame` | Caller's frame keeps `bool` dtype | `app.py` builds the frame from the request and reuses it for the response |
| `test_predict_rejects_a_non_dataframe` | `TypeError` | Fail at the boundary, not inside sklearn |
| `test_predict_requires_a_tracking_uri` | `RuntimeError` naming the variable | Without it the model can never load — fail loudly, not late |
| `test_load_model_requests_the_dev_alias_and_caches_per_model` | URI is `models:/<name>@dev`; two calls for one model load once | Pins the URI shape and proves the `lru_cache` works. The `@dev` alias is hardcoded (open item 3.2) — this test tells whoever makes it configurable what expectation they are changing |

---

## `test_api_endpoints.py` — 75 tests (14 functions)

Real routing, validation, response models, error mapping and background tasks; only
`predict` and `forward_to_monitoring` are replaced in `app.py`'s namespace.

Nine functions are parametrised over all five routes (`/predict_pk`, `/predict_cpk`,
`/predict_fk`, `/predict_cfk`, `/predict_normalform`).

| Test | ×N | Asserts | Why |
| :--- | ---: | :--- | :--- |
| `test_predict_route_returns_prediction_and_probability` | 5 | 200 + both output fields | Baseline happy path per route |
| `test_predict_route_echoes_every_request_feature` | 5 | Every request field reappears in the response | The response model extends the request model; a dropped feature would silently shrink the drift payload |
| `test_predict_route_calls_its_own_registered_model` | 5 | `pk_model`, `composite_pk_model`, `fk_model`, `composite_fk_model`, `denormalization_model` respectively | Catches a copy-paste slip between the five near-identical handlers |
| `test_predict_route_sends_all_declared_features_to_the_model` | 5 | Column list equals the Pydantic field order | The request-to-frame conversion must not reorder or drop |
| `test_predict_route_forwards_to_its_own_monitoring_track` | 5 | Exactly one forward, to `pk_columns`…`nf_columns` respectively, carrying features + prediction | Each model has its own reference and `DataDefinition`; a wrong track fails **inside** Evidently with "partially present in data" and is only visible in the dashboards |
| `test_predict_route_maps_a_model_failure_to_400` | 5 | Schema-drift `ValueError` → 400 with the detail preserved | A drift must be a client-visible 400, not a 500 |
| `test_predict_route_normalises_every_prediction_shape` | 30 | `np.int64`, `np.float64`, `list`, `tuple`, `np.ndarray`, `float` all become the right int | Covers the three-branch `.item()`/sequence/cast ladder that is duplicated in all five handlers |
| `test_predict_route_rejects_an_incomplete_body` | 5 | 422 **and** the model was never called | Validation must precede inference |
| `test_predict_route_rejects_an_uncoercible_value` | 5 | 422, model never called | Same for a wrongly-typed feature |
| `test_root_route_reports_the_service_is_alive` | 1 | 200 with a message | The liveness route the container check uses |
| `test_metrics_route_exposes_the_families_grafana_queries` | 1 | `http_requests_total`, `http_request_duration_seconds_bucket`, `http_request_duration_highr_seconds_bucket` are all present | An instrumentator upgrade that renames a family would leave every Golden-Signals panel blank and every alert permanently inactive — both silent failures |
| `test_new_data_event_accepts_and_publishes` | 1 | 202, event published, response JSON key is `schema` | `schema` is reserved on `BaseModel`, so the field is `schema_name` internally with a serialization alias |
| `test_new_data_event_defaults_the_schema` | 1 | Omitted schema defaults to `new_predict_data`, note passed through | The documented default of the streaming trigger |
| `test_new_data_event_returns_503_and_points_at_the_cron_fallback` | 1 | 503 whose detail mentions the scheduled run | A lost event is not lost data — the detail has to say so, or a caller escalates a non-incident |

---

## `test_data_models.py` — 39 tests (11 functions)

The API contract in both directions. Whether the declared features match the *trained*
models is a different question, answered against the live registry by
`test/test_models/test_model_schema_contract.py`.

Seven functions are parametrised over the five request/response model pairs.

| Test | ×N | Asserts | Why |
| :--- | ---: | :--- | :--- |
| `test_request_model_accepts_a_complete_payload` | 5 | A full payload validates and dumps to exactly the declared fields | Baseline |
| `test_request_model_declares_every_field_as_required` | 5 | **No feature has a default** | A silent default would be scored as if it were a real measurement — a wrong answer that looks valid, worse than a 422 |
| `test_request_model_rejects_a_missing_feature` | 5 | `ValidationError` naming the field | |
| `test_request_model_rejects_an_uncoercible_value` | 5 | `ValidationError` | Pydantic coerces `"5"` → `5`, so the test uses a genuinely uncoercible value |
| `test_request_model_ignores_unknown_fields` | 5 | Extras are **dropped, not rejected** | Documents current behaviour deliberately: `predict_batch` in the pipeline posts rows that still carry metadata columns, so switching to `extra="forbid"` would break it |
| `test_response_model_extends_the_request_model` | 5 | Subclass relation, and field order is request fields + `prediction` + `probability` | |
| `test_response_model_requires_the_model_output` | 5 | Constructing without `prediction` raises | |
| `test_notification_defaults_to_the_streaming_source_schema` | 1 | `new_predict_data`, `note=None` | |
| `test_notification_accepts_the_json_alias_and_the_field_name` | 1 | Both `schema` and `schema_name` work | `curl_tests/` posts the alias, Python callers use the field name |
| `test_notification_serializes_back_to_the_json_alias` | 1 | Dump uses `schema` | |
| `test_accepted_response_serializes_the_schema_alias` | 1 | Full response dict shape | |

---

## `test_monitoring_client.py` — 11 tests (9 functions) — 100% coverage

This module makes one promise: **monitoring can never fail a prediction.** It runs as a
FastAPI background task, so an escaping exception would be logged as an unhandled error
for a side effect nobody asked for — on every request, with the monitor down. Each
failure mode is asserted separately rather than trusted by inspection.

| Test | ×N | Asserts | Why |
| :--- | ---: | :--- | :--- |
| `test_forwards_the_payload_to_the_tracks_iterate_endpoint` | 1 | One POST to `{base}/iterate/{track}` with the payload | |
| `test_uses_the_configured_timeout` | 1 | Timeout is the configured value and ≤ 5s | A hung monitor must not tie up a worker thread per prediction |
| `test_each_track_gets_its_own_url` | 1 | All five tracks map to distinct URLs | Each model has its own feature schema, so the track is in the path, not the payload |
| `test_makes_no_request_when_monitoring_is_disabled` | 1 | `MONITORING_ENABLED=false` → zero calls | For running the API standalone with no monitor on the network |
| `test_swallows_request_failures` | 3 | `ConnectionError`, `Timeout`, generic `RequestException` all return `None` | The common case: monitor down or slow |
| `test_swallows_an_error_status` | 1 | 4xx/5xx via `raise_for_status` swallowed | |
| `test_swallows_exceptions_that_are_not_request_errors` | 1 | A `TypeError` is swallowed too | The bare `except Exception` is the actual safety net — unserialisable payload, exotic DNS error, typo in this module |
| `test_logs_a_transport_failure_at_debug_level` | 1 | Exactly one record, at DEBUG | With the monitor down this fires on every prediction; at warning level it would bury real errors |
| `test_logs_an_unexpected_failure_with_a_traceback` | 1 | One ERROR record with `exc_info` | The opposite choice: a non-transport error is a real defect — swallowed, but not hidden |

---

## `test_event_publisher.py` — 10 tests — 100% coverage

The push half of the streaming trigger, whose failure contract is the **opposite** of
`monitoring_client`: a lost event means the detector is not woken early, so this path
raises and the endpoint answers 503. (No data is lost either way — cron covers it.)

| Test | Asserts | Why |
| :--- | :--- | :--- |
| `test_posts_one_event_to_the_prefect_event_intake` | POST to `{PREFECT_API_URL}/events`, body is a **list** | The intake takes a list even for one event |
| `test_event_carries_the_names_the_automations_match_on` | Event name and resource id match the constants | These strings are duplicated in `prefect/change_events.py` (the webservice image cannot import it). If they drift, push triggers stop working and cron silently covers for them |
| `test_event_includes_id_and_occurred` | `id` parses as UUID, `occurred` is timezone-aware ISO | The Prefect *client* defaults these; the raw REST endpoint requires them and answers 422 without |
| `test_event_payload_carries_the_schema_and_the_note` | Resource and payload contents | Traceability of who triggered what |
| `test_a_missing_note_is_sent_as_an_empty_string` | `note=None` → `""` | The server rejects nulls here |
| `test_uses_the_configured_timeout` | Timeout is `EVENT_TIMEOUT_SECONDS` | |
| `test_a_transport_failure_raises_event_publish_error` | Raises, not swallows | This is what produces the endpoint's 503 |
| `test_the_error_names_the_endpoint_it_tried` | Message contains the API URL | First question when it fails is "which server?" |
| `test_the_error_includes_the_response_body` | A 422 body reaches the message | "422 Unprocessable Entity" alone does not say which field was rejected |
| `test_the_response_body_is_truncated` | Exactly 500 characters kept | A giant HTML error page must not be pasted whole into the 503 detail |

---

## `test_change_events.py` — 39 tests (19 functions) — 98% coverage

The concurrency contract of the streaming pipeline. Two consumers (`keys`, `nf`) work the
same rows independently and the trigger is at-least-once, so all of this is asserted
against `FakeConnection` — the interesting failures are in the SQL that gets built.

| Test | ×N | Asserts | Why |
| :--- | ---: | :--- | :--- |
| `test_unknown_track_is_rejected_before_any_sql_runs` | 20 | 4 functions × 5 bad tracks (`""`, `"KEYS"`, `"keys "`, an injection string, `"pk"`) all raise **and execute nothing** | The track is interpolated into the statement because column names cannot be bound — the allowlist is what makes that safe, and it must reject *before* executing |
| `test_known_tracks_are_accepted` | 2 | `keys` and `nf` work | The allowlist isn't over-tight |
| `test_recording_nothing_touches_the_database` | 1 | Empty change list → 0, no SQL | The detector calls this on every pass, including quiet ones |
| `test_recording_appends_one_insert_for_all_changes` | 1 | Exactly one INSERT with two value tuples | Round trips to Trino dominate the cost; a partial write would leave the work list inconsistent |
| `test_recording_creates_the_work_list_if_missing` | 1 | `CREATE SCHEMA` + `CREATE TABLE IF NOT EXISTS` | First run on a fresh install |
| `test_recording_binds_table_identifiers_instead_of_interpolating_them` | 1 | A table named `orders'); DROP TABLE staging--` never appears in the SQL text, only in params | Identifiers come from catalog metadata, but the loader controls those names — so they are untrusted input |
| `test_recording_stores_the_source_and_a_detection_timestamp` | 1 | `source`, `change_type`, plausible `detected_at` | Traceability of which detector found what |
| `test_claiming_marks_open_rows_then_reads_back_its_own` | 1 | UPDATE sets `<track>_claimed_by`, restricted to unclaimed+uncompleted; SELECT filters on `run_id` | The claim protocol itself |
| `test_claiming_deduplicates_tables_via_distinct` | 1 | `SELECT DISTINCT` | Duplicate change rows are intentionally not filtered on insert, so the reader collapses them — else a table reloaded five times is profiled five times |
| `test_claiming_uses_the_track_specific_columns` | 1 | Claiming `nf` touches no `keys_*` column | One track finishing must not hide the row from the other |
| `test_completing_only_closes_rows_this_run_claimed` | 1 | WHERE filters on `run_id`, returns rowcount | Changes detected *while* the run was in flight must stay open — the watermark has already moved on, so closing them drops them forever |
| `test_releasing_hands_the_work_back_without_completing_it` | 1 | Nulls the claim, does **not** set `completed_at` | Used when a run claims work then fails |
| `test_stale_claim_recovery_uses_a_cutoff_in_the_past` | 1 | Cutoff is ~`older_than_minutes` ago | Recovers work from a process killed outright, where the `finally` block never ran |
| `test_stale_claim_recovery_ignores_completed_and_unclaimed_rows` | 1 | WHERE requires claimed and not completed | Must not steal from healthy runs |
| `test_stale_claim_cutoff_is_string_comparable` | 1 | Cutoff matches `YYYY-MM-DD HH:MM:SS` | Timestamps are VARCHAR; lexicographic order matching chronological order is what makes the plain `<` valid |
| `test_counting_open_work_covers_every_track` | 1 | Condition includes both tracks, OR-ed | The detector emits its signal from this count. Missing a track would strand its work permanently: watermarks are persisted, so no later run would rediscover it |
| `test_counting_open_work_returns_zero_when_the_table_is_empty` | 1 | `scalar()` of `None` → 0 | Must not propagate `None` into arithmetic |
| `test_counting_open_work_creates_the_table_on_a_fresh_install` | 1 | Table is created | |
| `test_work_list_carries_one_claim_column_pair_per_track` | 1 | `<track>_claimed_by/_claimed_at/_completed_at` for every track | The schema that makes independent tracks possible |

---

## `test_change_detector.py` — 15 tests (12 functions)

`diff_snapshots` decides what gets reprocessed, so a wrong answer is either a missed
reload (stale predictions kept forever) or a permanent re-prediction loop. Pure function,
no Trino.

| Test | ×N | Asserts | Why |
| :--- | ---: | :--- | :--- |
| `test_first_run_reports_every_table_as_new` | 1 | No previous watermarks → all `new_table` | Cold start |
| `test_unchanged_tables_are_not_reported` | 1 | Identical counts → no changes | Prevents the re-prediction loop |
| `test_a_new_row_count_is_reported` | 1 | `row_count_changed` | Appends and reloads |
| `test_deleted_rows_are_reported_too` | 1 | Shrinking table also reported | A shrink is as much a reload as a growth |
| `test_a_column_change_outranks_a_row_change` | 1 | Both changed → `schema_changed` | The more informative label: the feature set itself may differ |
| `test_a_dropped_column_is_a_schema_change` | 1 | `schema_changed` | |
| `test_disappeared_tables_are_not_reported` | 1 | Vanished table yields nothing | Nothing left to predict on; existing predictions stay as history |
| `test_an_in_place_update_is_invisible` | 1 | Same counts, different values → **no change detected** | Records the documented blind spot: DuckDB over Trino has no WAL, no triggers, no `LISTEN/NOTIFY`. If someone later adds a checksum, this test tells them the contract changed on purpose |
| `test_tables_are_matched_on_the_full_three_part_name` | 1 | Same table name in two schemas stays distinct | Else loading `other_schema.customer` would mark `new_predict_data.customer` changed |
| `test_an_empty_snapshot_yields_no_changes` | 1 | Empty frame is safe | |
| `test_chunks_splits_without_losing_items` | 4 | Sizes 1, 2, 5, 10 over 5 items | Row counts are batched into UNION ALL queries; a lost chunk means a table is never counted and so never detected |
| `test_chunks_of_an_empty_list_is_empty` | 1 | No phantom empty batch | |

---

## `test_feature_recompute.py` — 16 tests

`recompute_table_stats` exists because features are extracted in 30-column batches while
several features are table-level. Computed per batch they would be wrong for any table
wider than one batch, so the SQL values are recomputed in pandas once all batches are in.

| Test | Asserts | Why |
| :--- | :--- | :--- |
| `test_table_level_counts_are_computed_across_all_batches` | Column count, unique count, row count, has-unique flags | The reason the function exists |
| `test_a_table_without_a_unique_column_is_flagged` | `table_has_no_single_pk_candidate=1`, no first-unique flag | This is what points the model at composite keys |
| `test_near_unique_columns_use_a_strict_threshold` | `> 0.95`, so exactly 0.95 does not count | Boundary that must match the training extractor |
| `test_other_unique_columns_excludes_the_column_itself` | `[1, 2, 1]` for two unique columns of three | **The subtle one.** Counting itself would shift the feature by one for exactly the rows the PK models care about most |
| `test_other_near_unique_columns_excludes_the_column_itself` | Same, for near-unique | |
| `test_only_the_first_unique_column_is_flagged` | `[1, 0, 0]` | Composite-key detection leans on this |
| `test_relative_ordinal_position_is_normalised_by_column_count` | 1/3, 2/3, 1.0 | |
| `test_unique_ratio_rank_orders_descending_with_ordinal_as_tiebreak` | `[2, 1, 3]` | Ties must resolve by position, as the SQL does |
| `test_unique_ratio_relative_to_max_scales_against_the_best_column` | 0.5, 1.0 | |
| `test_a_table_of_all_empty_columns_does_not_divide_by_zero` | 0.0, not NaN or a crash | An all-null table is a real input |
| `test_a_single_column_table_is_handled` | No off-by-one on a 1-column table | |
| `test_no_requeue_tables_produces_no_clause` | `("", {})` for `None` and `[]` | |
| `test_requeue_clause_binds_one_predicate_per_table` | One predicate + 3 params per table | Without this fragment a column is predicted once *ever*, so a reloaded table keeps its stale prediction |
| `test_requeue_clause_does_not_interpolate_table_names` | Hostile name only in params | |

### Defect found and fixed on 30.07.

| Test | Asserts | Why |
| :--- | :--- | :--- |
| `test_null_ratio_rank_ranks_by_null_ratio` | The column with no nulls ranks 1 and is the flagged one | Regression guard for the fix below |
| `test_null_ratio_rank_breaks_ties_by_ordinal_position` | Equal null ratios fall back to position | The SQL's second sort key |

The extraction SQL ranks by `null_ratio ASC, ordinal_position ASC`
([pk_fk_pipeline.py:136](../prefect/pk_fk_pipeline.py#L136)), but the Python recompute
that *overwrites* that value once all batches are in sorted by `ordinal_position` alone.
So `null_ratio_rank` meant "position in the table" and `is_least_null_in_table` meant "is
the first column", regardless of nulls — `null_ratio` was in the frame all along, the
sort key had simply been dropped. Fixed at
[pk_fk_pipeline.py:261](../prefect/pk_fk_pipeline.py#L261) by mirroring the SQL ordering.

Two things worth recording about it:

- **Only `pk_fk_pipeline.py` was affected.** `normalform_pipeline.py` already sorted by
  `["null_ratio", "ordinal_position"]` and was used as the reference implementation. An
  earlier note in `MLOPS_PLAN.md` claiming the bug was duplicated there was wrong.
- **The fix broke 11 tests**, all in this file. `recompute_table_stats` now reads
  `null_ratio`, and the `_feature_frame` helper had not been supplying it. The production
  frame always has the column (the feature query selects it), so the helper was the
  incomplete one — it now provides `null_ratio` by default. No fallback was added to the
  production code on purpose: a missing column should raise, not silently rank by
  something else, which is how this bug stayed invisible in the first place.

The retrain that follows this fix covers the four key models (pk/cpk/fk/cfk) and their
Evidently references; the normalform model consumed the correct value already.

---

## Coverage

Scope is defined in `[tool.coverage.run]`: the deployed artifact plus the pipeline
modules that have real unit tests. The training scripts and the SQL-heavy pipeline tasks
are excluded via `omit` — they need a live stack, so counting their ~2.5k statements
would only depress the number without saying anything about test quality.

| Module | Coverage |
| :--- | ---: |
| `webservice/predict.py` | 100% |
| `webservice/monitoring_client.py` | 100% |
| `webservice/event_publisher.py` | 100% |
| `webservice/data_model_*.py` (5 files) | 100% |
| `webservice/app.py` | 96% |
| `prefect/change_events.py` | 98% |
| `prefect/change_detector.py` | 37% |
| **Total** | **86%** |

Gate: `fail_under = 80` in `[tool.coverage.report]`, so it applies locally and in CI.
Plan item 2.2 asked for 70%.

`app.py`'s five uncovered lines are the `except HTTPException: raise` re-raises, which are
unreachable while the handler bodies raise nothing else. `change_detector.py`'s gap is its
Trino-touching tasks (`snapshot_source_tables`, `load_previous_watermarks`,
`persist_watermarks`, and the flow) — the pure diff logic around them is covered.

## Deliberately not covered here

| Gap | Why, and where it is covered instead |
| :--- | :--- |
| Real MLflow model loading | Needs the registry. `test/test_models/` covers the signature-vs-Pydantic contract against the live stack, marked `integration` |
| Real Trino queries | The extraction and queue SQL only run against DuckDB-over-Trino. Streaming has still never run end-to-end against real Trino data (open item 4.5) |
| Evidently report generation | Service boundary; the client contract is covered, the service's own behaviour is not |
| Prefect flow orchestration | Deployments, triggers and concurrency limits are configuration, verified by `scripts/setup_stack.sh` and the Prefect UI |
| `/metrics` under the container's instrumentator 7.1.0 | Tests run against 8.1.0 — 7.x requires `starlette<1.0.0`, which cannot coexist with prefect. The metric families were verified identical across both versions (see the comment in `pyproject.toml`) |
