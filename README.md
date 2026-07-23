# Alligator Metaswamp

**Predicting primary keys, foreign keys and normal forms from database metadata alone.**

Point this service at a database with no documented constraints and it suggests, for
each column, whether it is a primary key, a foreign key, and which normal form its table
satisfies — using only metadata and column statistics, never the row data itself.

This repository is the ML-engineering build around that idea: a tested Python package, a
containerised FastAPI service, an MLflow registry, CI, and Prometheus / Grafana /
Evidently monitoring.

---

## Why

Databases often arrive without documented keys — legacy systems, data-lake dumps,
third-party imports. The tables and columns are there, but the primary/foreign-key
relationships are not. Recovering them by hand means writing SQL against every table to
find unique, non-null columns, then eyeballing column names across databases to guess the
references. It is slow, subjective, and keys get missed.

The typical downstream need is data modelling — especially Data Vault, where the first
step is deciding which columns are the business keys for Hubs, Satellites and Links. This
service turns "guess the keys" into a classification problem and hands a data engineer a
ranked, probability-scored list of candidates to confirm.

Because inference uses only `information_schema`-style metadata and cheap aggregate
statistics, it never reads real data — which keeps it fast and usable on regulated data
(medical, financial) without a data-transfer discussion.

---

## The four tasks

| Task | Question | Target | Status |
| :--- | :--- | :--- | :--- |
| **1. Primary keys** | Is this column a single PK? part of a composite PK? | `pk_target`, `composite_pk_target` | ✅ implemented |
| **2. Foreign keys** | Is this column a single FK? part of a composite FK? | `fk_target`, `composite_fk_target` | ✅ implemented |
| **3. Normalization** | What is the highest normal form the table satisfies (0–3NF)? | `target_normal_form` | ✅ implemented |
| **4. Domain grouping** | Group tables/schemas by naming and structural similarity. | — (unsupervised) | ⛔ out of scope |

> Task 4 is deliberately **not** part of this capstone. Effort is going into engineering
> depth on Tasks 1–3, per the capstone brief ("keep the modelling simple; the point is
> the engineering").

Each task is a separate model, so five models are trained and served in total:
`pk_model`, `composite_pk_model`, `fk_model`, `composite_fk_model`, `denormalization_model`.

---

## Architecture

```
                            TRAINING (offline)
   ┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │ Source DBs │──▶│  training scripts  │──▶│   MLflow Registry    │
   │ Trino / CSV│   │  5 models (sklearn)│   │  versions + @alias   │
   └────────────┘   └────────────────────┘   └──────────┬───────────┘
                                                         │ resolve models:/name@alias
                            SERVING (online)             ▼
   ┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │  client    │──▶│  FastAPI service   │──▶│  model held in memory │
   │ REST / curl│   │  5 typed endpoints │   │  (cached, per model)  │
   └────────────┘   └─────────┬──────────┘   └──────────────────────┘
                              │ /metrics, prediction events
                              ▼
             ┌──────────────────────────────────────────┐
             │ Prometheus ── Grafana   ·   Evidently     │
             │ golden signals + alerts   input drift     │
             └──────────────────────────────────────────┘
```

Everything runs from a single `docker-compose.yaml`. Training and serving are decoupled
through the registry: promoting a model is moving an alias, not rebuilding a container.

---

## Quickstart

### 1. Environment

The project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```bash
# install dependencies (runtime + dev) into a local venv
uv sync --extra dev
# or, with plain pip:
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Copy the environment template and fill in the values:

```bash
cp env.template .env
```

| Variable | Purpose |
| :--- | :--- |
| `MLFLOW_TRACKING_URI` | Where the service and training scripts reach MLflow (e.g. `http://127.0.0.1:5000`). **Required.** |
| `MLFLOW_MODEL_ALIAS` | Which registry alias the service serves. Defaults to `dev`. |
| `TRINO_USERNAME` / `TRINO_PASSWORD` / `TRINO_IP_ADDRESS` | Trino connection for pulling live metadata (optional; training also works from the bundled CSVs). |

### 2. Train and register the models

Start an MLflow server, then run the training scripts (each logs a run and registers a
model under the `dev` alias):

```bash
mlflow server --host 0.0.0.0 --port 5000        # in one terminal
python task_1/task_1_pk_train_and_register.py    # single primary key
python task_1/task_1_cpk_train_and_register.py   # composite primary key
python task_2/task_2_fk_train_and_register.py    # single foreign key
python task_2/task_2_cfk_train_and_register.py   # composite foreign key
python task_3/task_3_denormalization_train_and_register.py  # normal form
```

### 3. Run the full stack

```bash
docker compose up --build
```

| Service | URL |
| :--- | :--- |
| Prediction API (Swagger docs) | http://localhost:8080/docs |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard auto-provisioned) | http://localhost:3000 |
| Evidently drift report | http://localhost:8085/report |

### 4. Make a prediction

```bash
bash curl_tests/test_curl_predict_pk.sh
```

or directly:

```bash
curl -X POST http://localhost:8080/predict_pk \
  -H "Content-Type: application/json" \
  -d '{ "number_unique_values": 768, "count": 768, "is_unique": 1, ... }'
```

Response echoes the input features and appends the model output:

```json
{ "...": "...", "prediction": 1, "probability": 0.94 }
```

`prediction` is the predicted class; `probability` is the model's confidence in **that**
class (so for the 4-class normal-form model it is the confidence in the returned normal
form, not P(class=1)).

---

## API

| Method | Route | Model | Returns |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | — | liveness message |
| `GET` | `/health` | — | which models resolved (`ok` / `degraded`) |
| `GET` | `/metrics` | — | Prometheus scrape target |
| `POST` | `/predict_pk` | `pk_model` | single primary key |
| `POST` | `/predict_cpk` | `composite_pk_model` | composite primary key |
| `POST` | `/predict_fk` | `fk_model` | single foreign key |
| `POST` | `/predict_cfk` | `composite_fk_model` | composite foreign key |
| `POST` | `/predict_normalform` | `denormalization_model` | normal form 0–3 |

The request schema for each route is a Pydantic model in `webservice/data_model_*.py` —
that file is the source of truth for the exact feature list.

---

## Data

Training data is metadata + statistics extracted from 60+ publicly available and
synthetically generated databases (Kaggle, TPC-H, Northwind, Spider, Sakila, Chinook,
MIMIC-IV, plus Python- and LLM-generated synthetic schemas). Only metadata is used, so
the datasets can be small and varied without any real records being stored.

| File | Rows (columns) | Databases | Tables | Used by |
| :--- | ---: | ---: | ---: | :--- |
| `data/summary_output_task_1_2_training.csv` | 10,348 | 241 | 1,706 | Tasks 1 & 2 |
| `data/nf_test_analyse.csv` | 13,822 | 503 | 1,560 | Task 3 |
| `data/raw_metadata.csv` | 3,009 | 54 | 466 | raw extract / EDA |

### Class balance

Keys are rare by construction (a table has one PK and many ordinary columns), so the
targets are imbalanced — **accuracy is meaningless here; we track F1 and PR-AUC instead.**

| Task 1 / 2 target | Positive rate | | Task 3 class | Share |
| :--- | ---: | --- | :--- | ---: |
| Single primary key | 12.0 % | | 0NF (violates 1NF) | 22.8 % |
| Composite primary key | 8.7 % | | 1NF | 23.7 % |
| Single foreign key | 16.5 % | | 2NF | 22.1 % |
| Composite foreign key | 2.1 % | | 3NF | 31.4 % |

Task 3 data is synthetically generated to keep the four classes balanced — a luxury the
key tasks do not have.

---

## Features

Features are engineered per column and per table and split into two sets, generated
directly from the training scripts:

**Tasks 1 & 2 — 31 features** (identical feature set; only the target differs)

Column-level: `number_unique_values`, `count`, `is_unique`, `ordinal_position`,
`unique_ratio`, `is_first_column`, `relative_ordinal_position`, `is_first_unique_column`,
`unique_ratio_rank`, `null_ratio_rank`, `is_least_null_in_table`,
`unique_ratio_relative_to_max`, `name_ends_with_id`, `name_contains_table_name`,
`name_is_singular_table_id`, `name_length`, and one-hot `column_type_{boolean, date,
decimal, double, integer, varchar}`.

Table-level: `table_column_count`, `table_unique_column_count`, `table_row_count`,
`table_has_unique_column`, `table_has_no_single_pk_candidate`,
`table_near_unique_column_count`, `table_id_named_column_count`,
`table_non_null_column_count`, `table_max_unique_ratio`.

**Task 3 — 50 features**

Everything above (with `char`/`timestamp` one-hot types added), plus normalization-specific
features: `is_this_col_violating_1nf`, `is_composite_key_part`,
`is_this_col_partial_dependency`, `table_has_composite_pk`, `table_has_partial_dependency`,
`table_avg_unique_ratio`, `table_avg_null_ratio`, `table_std_unique_ratio`,
`table_ratio_of_pk_candidates`, `table_ratio_composite_key_cols`,
`table_ratio_1nf_violations`, plus `null_count`, `null_ratio`, `is_non_null` and the
`other_*` counts (which Tasks 1 & 2 drop).

> **Note on Task 3 features.** `is_this_col_violating_1nf`, `is_composite_key_part` and
> `is_this_col_partial_dependency` are close to the *definitions* of 1NF/2NF, so the model
> partly learns from labels derived the same way the target is. `target_normal_form` and
> `table_contains_1nf_violation` are dropped from the features to avoid direct leakage.
> This is a known limitation we call out rather than hide.

Identifier columns (`database`, `schema`, `table_name`, `column_name`) and raw
`min_value` / `max_value` are never used as features — only for grouping and traceability.

---

## Model & results

Every task uses a **RandomForestClassifier** with `class_weight="balanced"` and otherwise
default hyperparameters. The train/test split is **grouped by database** (`StratifiedGroupKFold`),
so columns from one database never appear on both sides — otherwise the model memorises a
naming convention and the test score lies.

Example F1 scores from the registry:

| Model | Train F1 | Test F1 |
| :--- | ---: | ---: |
| `composite_pk_model` | 0.978 | 0.712 |
| `composite_fk_model` | 1.000 | 0.811 |

The high train F1 is an un-tuned RandomForest memorising the training set. Tuning it is a
non-goal — the graded surface is the engineering. Instead, the plan is a CI **model-quality
gate** that refuses to register a model whose test F1 falls below the current baseline.

---

## MLOps stack

| Concern | Tool | Status |
| :--- | :--- | :--- |
| Experiment tracking & registry | MLflow | ✅ params, metrics, signature, feature list; alias-based deploy |
| Model service | FastAPI + Docker | ✅ 5 typed endpoints, `/health`, `/metrics` |
| CI | GitHub Actions | ✅ Ruff lint/format + pytest with coverage |
| Service monitoring | Prometheus + Grafana | ✅ golden signals, 11 alert rules, provisioned dashboard |
| Model monitoring | Evidently | ✅ input drift against a real reference set |
| Data pipeline | Prefect + dbt | 🔜 planned |
| Retraining | manual trigger | 🔜 planned (`reload_models()` hook in place) |

### Monitoring detail

- `webservice/metrics.py` adds model-level metrics on top of the HTTP golden signals:
  predictions by class, inference duration, errors by type, returned confidence.
- `prometheus/alert.yaml` covers latency, traffic, errors, saturation, plus model-health
  rules (a model stuck on one class, confidence collapse, prediction errors spiking).
- `grafana/dashboards/golden-signals.json` is auto-provisioned — a fresh `docker compose
  up` shows the board with no manual setup.
- `evidently_service/build_reference.py` regenerates the drift reference set from the
  training data. **Re-run it whenever the feature set changes.**

---

## Testing

```bash
pytest                              # unit + API tests, with coverage
pytest --cov-report=term-missing    # see uncovered lines
ruff check . && ruff format --check .
```

`test/test_api/` covers the API surface (routing, validation, error mapping, `/metrics`,
`/health`) and the prediction helper (including a regression test for the multi-class
probability bug). `test/test_data/` validates the training CSVs.

---

## Project structure

```
webservice/            FastAPI app, Pydantic schemas, predict + metrics
task_1/ task_2/ task_3/  training + registration scripts, one per model
evidently_service/     drift-monitoring service + reference builder
prometheus/            scrape config + alert rules
grafana/               provisioned datasource + dashboard
test/                  API, prediction and data-quality tests
data/                  training CSVs
documentation/         MLOps plan, presentations
```

---

## Tech stack

Python · scikit-learn · FastAPI · Pydantic · MLflow · Docker Compose · Trino ·
Prometheus · Grafana · Evidently · Ruff · pytest · GitHub Actions · uv

---

## Contributors

Christian · Niklas · Nijat
