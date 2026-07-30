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
# install dependencies into a local venv - test tooling included, no extra flag needed
uv sync
# or, with plain pip:
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e .
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

<details>
<summary><b>Source databases</b> (click to expand)</summary>

Public datasets, sample databases, and Python/LLM-generated synthetic schemas. "Number of
tables" is the size of each source database, not the number of feature rows.

| Dataset | Where to find | Tables | Topic |
| :--- | :--- | ---: | :--- |
| Willibald DWA Challenge | https://dwa-compare.info/en/start-2/ | 10 | E-Commerce |
| Synthetic E-Commerce Dataset — Free Sample | https://www.kaggle.com/datasets/oreomonsta123/synthetic-e-commerce-dataset10-tables-8-countries | 10 | E-Commerce |
| Multitable Ecommerce European Fashion | https://www.kaggle.com/datasets/joycemara/european-fashion-store-multitable-dataset | 7 | E-Commerce |
| TPC-H | https://www.tpc.org/tpch/ | 8 | E-Commerce |
| Formula 1 World Championship 2000–2026 | https://www.kaggle.com/datasets/mkaur1141/formula-1-world-championship-dataset-20002026 | 5 | Sports |
| USDA Nutrition Data: Flattened (SR Legacy) | https://www.kaggle.com/datasets/ericfornow/usda-nutrition-data-flattened-sr-legacy | 7 | Science |
| Solstice Residential Energy Pack (Sample) | https://www.kaggle.com/datasets/justinsolstice/solstice-residential-energy-pack | 16 | Energy |
| Northwind | https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs | 11 | E-Commerce |
| House Sales in King County, USA | https://www.kaggle.com/datasets/harlfoxem/housesalesprediction | 2 | Finance |
| Take me home | neue-fische database | 1 | unknown |
| Petsowners | neue-fische database | 4 | Veterinary |
| Monalisa | neue-fische database | 6 | Aviation |
| Heart | neue-fische database | 5 | Medicine |
| Gapminder | neue-fische database | 5 | Medicine |
| Bike Store relational database | https://www.kaggle.com/datasets/dillonmyrick/bike-store-sample-database | 9 | E-Commerce |
| LifeAlly (career, health, relationship, finance) | https://www.kaggle.com/datasets/karanshelar6/lifeallycarrer-health-relationship-finance | 7 | Various |
| Sales, customers, products | https://www.kaggle.com/datasets/arkhepacis/sales-customers-products-and-customers | 5 | E-Commerce |
| F1 Grand Prix Dataset | https://www.kaggle.com/datasets/harshitstark/f1-grandprix-datavault | 14 | Sports |
| FIFA World Cup Database | https://www.kaggle.com/datasets/joshfjelstul/world-cup-database | 27 | Sports |
| Northwind Datavault | Microsoft Northwind Database | 120 | E-Commerce |
| Euroleague & Eurocup Datasets | https://www.kaggle.com/datasets/babissamothrakis/euroleague-datasets | 14 | Sports |
| mimic-iv-clinical-database-demo-2.2 | https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2 | 31 | Medicine |
| Academic Records | Python-generated synthetic | 10 | Education |
| Aviation Operations | Python-generated synthetic | 6 | Aviation |
| Banking System | Python-generated synthetic | 9 | Finance |
| Calculated Field Examples | Python-generated synthetic | 4 | Various |
| Clinical Trials | Python-generated synthetic | 10 | Medicine |
| Content Management Platform | Python-generated synthetic | 10 | Entertainment |
| Diabetes Dataset | Python-generated synthetic | 4 | Medicine |
| E-Commerce Operations | Python-generated synthetic | 9 | E-Commerce |
| Event Management | Python-generated synthetic | 11 | Entertainment |
| Fleet Management | Python-generated synthetic | 10 | Transportation |
| Food Delivery | Python-generated synthetic | 10 | Food & Beverage |
| Government Database | Python-generated synthetic | 6 | Government |
| Healthcare Database | Python-generated synthetic | 7 | Healthcare |
| Hospitality Database | Python-generated synthetic | 10 | Hospitality |
| HR System | Python-generated synthetic | 9 | Human Resources |
| Insurance Database | Python-generated synthetic | 10 | Finance |
| Introduction Database | Python-generated synthetic | 3 | Various |
| Inventory Management | Python-generated synthetic | 9 | Logistics |
| IoT Platform | Python-generated synthetic | 10 | Technology |
| Library Database | Python-generated synthetic | 9 | Education |
| Manufacturing Database | Python-generated synthetic | 10 | Manufacturing |
| Media Streaming Platform | Python-generated synthetic | 10 | Entertainment |
| Project Management | Python-generated synthetic | 9 | Various |
| Railway Database | Python-generated synthetic | 6 | Transportation |
| Real Estate Database | Python-generated synthetic | 10 | Finance |
| Retail Data Warehouse | Python-generated synthetic | 6 | Retail |
| Retail Point of Sale | Python-generated synthetic | 10 | Retail |
| Shipping Database | Python-generated synthetic | 6 | Logistics |
| Social Platform | Python-generated synthetic | 8 | Technology |
| Sports League Database | Python-generated synthetic | 10 | Sports |
| Supply Chain | Python-generated synthetic | 10 | Logistics |
| Telecom Billing | Python-generated synthetic | 9 | Telecommunications |
| Telehealth Platform | Python-generated synthetic | 10 | Healthcare |
| Traffic Management | Python-generated synthetic | 6 | Transportation |
| University Database | Python-generated synthetic | 6 | Education |
| Synthetic Multi-Domain Collection | Python-generated synthetic | 87 | Various |
| FreeSQL.com Sample Databases | https://freesql.com | 62 | Various |
| Spider Text-to-SQL Benchmark | https://yale-lily.github.io/spider | ~1,020 (~190 DBs) | Various (138 domains) |
| Chinook Sample Database | https://github.com/lerocha/chinook-database | 11 | Music |
| Sakila Sample Database | https://github.com/jOOQ/sakila | 16 | Video rental |

</details>

---

## Features

Features are engineered per column and per table. There are two feature sets, generated
directly from the training scripts:

- **Tasks 1 & 2 — 31 features** (identical set; only the target differs)
- **Task 3 — 50 features** (adds normalization-specific signals and a few columns Tasks 1
  & 2 drop)

Identifier columns (`database`, `schema`, `table_name`, `column_name`) and raw
`min_value` / `max_value` are never used as features — only for grouping and traceability.

> **Note on Task 3 features.** `is_this_col_violating_1nf`, `is_composite_key_part` and
> `is_this_col_partial_dependency` are close to the *definitions* of 1NF/2NF, so the model
> partly learns from labels derived the same way the target is. `target_normal_form` and
> `table_contains_1nf_violation` are dropped from the features to avoid direct leakage.
> This is a known limitation we call out rather than hide.

<details>
<summary><b>Full metadata &amp; statistics dictionary</b> (click to expand)</summary>

`T1` = single/composite primary key · `T2` = single/composite foreign key ·
`T3` = normal form · `identifier` = used for grouping only · `—` = present in the data
but not used as a training feature.

| Variable | Description | Example | Used in |
| :--- | :--- | :--- | :--- |
| `database` | Name of the database the table lives in | "Formula 1" | identifier |
| `schema` | Name of the schema the table lives in | "Season 2025" | identifier |
| `table_name` | Name of the table | "Drivers" | identifier |
| `column_name` | Name of the column | "Age" | identifier |
| `column_type` | Raw data type (one-hot encoded for training) | "int" | — |
| `min_value` | Minimum value of the column | "18" | — |
| `max_value` | Maximum value of the column | "44" | — |
| `number_unique_values` | Count of distinct values in the column | "23" | T1, T2, T3 |
| `count` | Number of rows in the table | "25" | T1, T2, T3 |
| `null_count` | Raw count of null values in the column | "0" | T3 |
| `null_ratio` | `null_count / count` | "0.0" | T3 |
| `is_unique` | Column values are fully unique | 0 or 1 | T1, T2, T3 |
| `ordinal_position` | Position of the column in the table (1-based) | "2" | T1, T2, T3 |
| `unique_ratio` | `number_unique_values / count` | "0.177" | T1, T2, T3 |
| `is_non_null` | Column has no null values | 0 or 1 | T3 |
| `is_first_column` | Column is the first in the table | 0 or 1 | T1, T2, T3 |
| `relative_ordinal_position` | `ordinal_position / table_column_count` | "0.222" | T1, T2, T3 |
| `is_first_unique_column` | Column is the first unique column in the table | 0 or 1 | T1, T2, T3 |
| `table_column_count` | Total columns in the table | "9" | T1, T2, T3 |
| `table_unique_column_count` | Number of fully unique columns in the table | "0" | T1, T2, T3 |
| `table_row_count` | Number of rows in the table | "768" | T1, T2, T3 |
| `other_unique_columns_in_table` | Count of *other* unique columns in the table | "0" | T3 |
| `table_has_unique_column` | Table has at least one unique column | 0 or 1 | T1, T2, T3 |
| `table_has_no_single_pk_candidate` | No single-column PK candidate exists | 0 or 1 | T1, T2, T3 |
| `table_near_unique_column_count` | Number of near-unique columns in the table | "0" | T1, T2, T3 |
| `table_id_named_column_count` | Number of ID-named columns in the table | "0" | T1, T2, T3 |
| `table_non_null_column_count` | Number of non-null columns in the table | "9" | T1, T2, T3 |
| `table_max_unique_ratio` | Highest `unique_ratio` in the table | "0.671" | T1, T2, T3 |
| `table_integer_column_count` | Number of integer-typed columns in the table | "7" | T3 |
| `unique_ratio_rank` | Rank of this column's `unique_ratio` in the table | "4" | T1, T2, T3 |
| `null_ratio_rank` | Rank of this column's `null_ratio` in the table | "2" | T1, T2, T3 |
| `is_least_null_in_table` | Column has the lowest `null_ratio` in the table | 0 or 1 | T1, T2, T3 |
| `unique_ratio_relative_to_max` | `unique_ratio / table_max_unique_ratio` | "0.264" | T1, T2, T3 |
| `other_near_unique_columns_in_table` | Count of *other* near-unique columns | "0" | T3 |
| `name_ends_with_id` | Column name ends with "id" | 0 or 1 | T1, T2, T3 |
| `name_contains_key` | Column name contains "key" | 0 or 1 | T3 |
| `name_contains_table_name` | Column name contains the table name | 0 or 1 | T1, T2, T3 |
| `name_is_singular_table_id` | Column name = singular table name + "id" | 0 or 1 | T1, T2, T3 |
| `name_length` | Length of the column name | "7" | T1, T2, T3 |
| `column_type_boolean` | One-hot: boolean | 0 or 1 | T1, T2 |
| `column_type_char` | One-hot: char | 0 or 1 | T3 |
| `column_type_date` | One-hot: date | 0 or 1 | T1, T2, T3 |
| `column_type_decimal` | One-hot: decimal | 0 or 1 | T1, T2, T3 |
| `column_type_double` | One-hot: double | 0 or 1 | T1, T2, T3 |
| `column_type_integer` | One-hot: integer | 0 or 1 | T1, T2, T3 |
| `column_type_timestamp` | One-hot: timestamp | 0 or 1 | T3 |
| `column_type_varchar` | One-hot: varchar | 0 or 1 | T1, T2, T3 |
| `is_this_col_violating_1nf` | Column has non-atomic / multi-valued entries | 0 or 1 | T3 |
| `is_composite_key_part` | Column is part of a composite primary key | 0 or 1 | T3 |
| `is_this_col_partial_dependency` | Column partially depends on the composite PK | 0 or 1 | T3 |
| `table_avg_unique_ratio` | Mean `unique_ratio` across the table's columns | "0.312" | T3 |
| `table_avg_null_ratio` | Mean `null_ratio` across the table's columns | "0.05" | T3 |
| `table_std_unique_ratio` | Std dev of `unique_ratio` across the table | "0.21" | T3 |
| `table_ratio_of_pk_candidates` | Ratio of PK-candidate columns to total columns | "0.11" | T3 |
| `table_has_composite_pk` | Table uses a composite primary key | 0 or 1 | T3 |
| `table_ratio_composite_key_cols` | Ratio of composite-key columns to total | "0.22" | T3 |
| `table_ratio_1nf_violations` | Ratio of 1NF-violating columns to total | "0.0" | T3 |
| `table_has_partial_dependency` | Any column in the table is a partial dependency | 0 or 1 | T3 |
| `table_contains_1nf_violation` | Table has a 1NF violation | 0 or 1 | — (dropped, leakage guard) |
| `pk_target` | Is the column a single primary key? | 0 or 1 | **target: T1** |
| `composite_pk_target` | Is the column part of a composite PK? | 0 or 1 | **target: T1** |
| `fk_target` | Is the column a foreign key? | 0 or 1 | **target: T2** |
| `composite_fk_target` | Is the column part of a composite FK? | 0 or 1 | **target: T2** |
| `target_normal_form` | Highest normal form the table satisfies (0–3) | 0, 1, 2, 3 | **target: T3** |

</details>

### Data types

The current pipeline covers standard scalar types. Future work should add types like
`blob`, while intentionally leaving out semi-structured data such as `json` and `xml`.

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
