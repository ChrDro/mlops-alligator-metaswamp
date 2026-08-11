# Alligator Metaswamp

Say goodbye to manual database archaeology. Automatically discover primary and foreign keys, normalization forms, and data domains using only metadata and statistics—privacy-safe, automated, effortless.

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
| **4. Domain grouping** | Which business domain (subject area) does this table belong to? | `subject_area` (discovered, not predefined) | ✅ implemented |

> Task 4 has no ground truth, so it runs in two stages: subject areas are **discovered**
> by clustering table and column names (embeddings → UMAP → HDBSCAN) and **named** by a
> local open-weights LLM, then that labelling is distilled into a TF-IDF text classifier
> which is what gets registered and served. Retraining can rename an area or add a new
> one, so its label set is not fixed in code.

Each task is a separate model, so six models are trained and served in total:
`pk_model`, `composite_pk_model`, `fk_model`, `composite_fk_model`,
`denormalization_model`, `subject_area_model`.

---

## Architecture

```
                            TRAINING (offline)
   ┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │ Source DBs │──▶│  training scripts  │──▶│   MLflow Registry    │
   │ Trino / CSV│   │  6 models          │   │  versions + @alias   │
   └────────────┘   └────────────────────┘   └──────────┬───────────┘
                                                         │ resolve models:/name@alias
                            SERVING (online)             ▼
   ┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │  client    │──▶│  FastAPI service   │──▶│  model held in memory │
   │ REST / curl│   │  6 typed endpoints │   │  (cached, per model)  │
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
cp .env.example .env
```

| Variable | Purpose |
| :--- | :--- |
| `MLFLOW_TRACKING_URI` | Where the service and training scripts reach MLflow (e.g. `http://127.0.0.1:5000`). **Required.** |
| `MODEL_ALIAS` | Which registry alias the running model-service resolves each model from. Defaults to `dev`; `docker-compose.yaml` sets it to `prod`, so the full stack needs a model promoted to `prod` before predictions work (see [step 2b](#2b-promote-a-model-to-prod)). |
| `TRINO_USERNAME` / `TRINO_PASSWORD` / `TRINO_IP_ADDRESS` | Trino connection for pulling live metadata (optional; training also works from the bundled CSVs). `TRINO_IP_ADDRESS` is a host without a port - use `localhost` for the Trino container from `docker-compose.yaml`. The user is created in `password.db` by the credential script below. |
| `TRINO_KEYSTORE_PASSWORD` / `TRINO_SHARED_SECRET` | Protect the generated TLS keystore and Trino's internal communication. Any non-empty values work locally. |
| `OLLAMA_MODEL` / `OLLAMA_MODEL_FAMILY` | Which open-weights model names the task_4 subject areas. Defaults to `qwen2.5:3b`; use `qwen2.5:7b` if Docker has ≥8 GB RAM. Keep the two in sync — the container healthcheck greps for the family. |
| `OLLAMA_HOST_PORT` / `OLLAMA_BASE_URL` | Set both to a free port if the host already runs Ollama natively on 11434. |


Then generate Trino's credentials. It serves HTTPS with password auth, and the two
files that needs — a TLS keystore holding a private key and a bcrypt password file —
are git-ignored, so a clone has neither:

```bash
bash trino-iceberg/generate-dev-credentials.sh
```

Run this **before** the first `docker compose up`: Compose bind-mounts both paths, and
Docker creates a directory for a bind-mount source that is missing, which leaves Trino
unable to start. The script is idempotent, keeps existing files, and takes `--force` to
rotate them. `scripts/setup_stack.sh` runs it in preflight, so the scripted path needs
no extra step. See [trino-iceberg/README.md](trino-iceberg/README.md) for details.

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
# Subject area. Needs the ollama service up (docker compose up -d ollama) to name the
# clusters it discovers, and downloads a sentence-transformer on first run.
python task_4/task_4_subject_area_train_and_register.py     # subject area
```

The single-FK script takes **around 9 minutes**, noticeably longer than the others: it
runs three hyperparameter searches, scores six candidates over five grouped folds each,
and trains on three reference-scope widths (see [Features](#features)), so ~25k rows per
fit rather than ~8k. During each search scikit-learn prints `Fitting 5 folds for each of
30 candidates` and then stays quiet for minutes — that is the expected behaviour, not a
hang. Piping the output hides progress entirely because Python buffers stdout, so use
`python -u` if you redirect to a file.

### 2b. Promote a model to `prod`

Training always registers under the `dev` alias, but the model-service in
`docker-compose.yaml` serves `@prod` (see the `MODEL_ALIAS` table above). Move the alias
with:

```bash
python scripts/promote_model.py                  # all 6 models
python scripts/promote_model.py --model pk_model  # just one
python scripts/promote_model.py --dry-run         # print the decision, move nothing
```

It only moves `prod` to the `dev` candidate when its metric (`test_f1_score` by default,
override with `--metric`) is at least as good as the model currently serving `prod` — or
unconditionally the first time, when `prod` does not exist yet. Safe to re-run: if `dev`
and `prod` already point at the same version, it is a no-op.

### 2c. Restart the model service after promoting

```bash
docker compose restart model-service    # only if the stack is already running
```

**This is not optional after a promotion.** The service resolves each model once and
caches it with `@lru_cache`, so a freshly promoted version is ignored until the process
restarts. Skipping it produces a confusing failure: the registry holds the new model, but
every request is scored by the old one, and any request carrying new features comes back
as `400 Feature mismatch … the model expects N features`. The
[contract tests](#testing) report the same drift for the same reason.

Whenever the **feature set** changed, also rebuild the monitoring baselines and their
image — a baseline describes one specific model version, so it is stale the moment that
version changes:

```bash
python evidently_service/build_monitoring_references.py --only fk_columns
docker compose up -d --build evidently_service
```

### 3. Run the full stack

```bash
# Trino's keystore and password file are git-ignored - generate them first (step 1).
# Skipping this leaves Trino unable to start: Compose bind-mounts the two missing
# paths and Docker creates directories there. Re-running the script clears them.
bash trino-iceberg/generate-dev-credentials.sh

docker compose up --build
```

Or let `scripts/setup_stack.sh` do all of it — it runs the generator in preflight and
takes the stack from a fresh clone to a working prediction in one command.

| Service | URL |
| :--- | :--- |
| Prediction API (Swagger docs) | http://localhost:8080/docs |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard auto-provisioned) | http://localhost:3000 |
| Evidently drift report | http://localhost:8085/report |
| Ollama (task_4 subject-area labels) | http://localhost:11434 |

On first start the `ollama` service pulls its model (~2 GB for the default
`qwen2.5:3b`) into a named volume, so the container reports `starting` for a few
minutes before it turns `healthy`. Every later `up` reuses the volume. Label generation
needs no API key and no internet access.
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4200 |
| MinIO console | http://localhost:9001 |
| Trino (self-signed TLS, password auth) | https://localhost:8443 |
| Nessie (Iceberg catalog) API | http://localhost:19120/api/v1 |

Trino, Nessie and the `warehouse` bucket are part of this stack, sharing the same
`minio` service that MLflow stores artifacts in. Catalogs: `iceberg` (Nessie on
MinIO), `duckdb` (`trino-iceberg/data/capstone.db`) and `tpch`.

#### Alternative: start the stack from a container (Docker-out-of-Docker)

```bash
./scripts/run_stack_in_container.sh
```

This runs the compose CLI inside a container that has the host's Docker socket
mounted, so the services it starts are siblings of the launcher on the host
daemon — same published ports, same named volumes, same image cache as
`docker compose up`. There is no nested Docker daemon and no `privileged: true`.
Useful when the machine that orchestrates the stack should not need a Python
environment or a compose plugin of its own, only a socket.

Arguments are passed straight through to `docker compose` inside the launcher:

```bash
./scripts/run_stack_in_container.sh ps
./scripts/run_stack_in_container.sh logs -f grafana
./scripts/run_stack_in_container.sh up -d prometheus grafana
./scripts/run_stack_in_container.sh down
```

Works on macOS, Linux and Windows (Git Bash or WSL2) with Docker Desktop. The
launcher has to mount the repo at the **same absolute path the Docker daemon
uses for it** — the host path on macOS and Linux, `/mnt/<drive>/...` under
Docker Desktop for Windows: the relative bind mounts in `docker-compose.yaml`
are resolved by the CLI in the launcher but mounted by the host daemon, so a
wrong path would silently start Prometheus and Grafana without their
configuration. The script probes this before starting anything and aborts with
the detected path if the daemon cannot see the repo; override it with
`PROJECT_DIR=... ./scripts/run_stack_in_container.sh` for an unusual setup.

Mounting the Docker socket is equivalent to root on the host — this is meant for
local development and CI, not for running untrusted code.

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
| `GET` | `/health/live` | — | `alive` as soon as the process serves HTTP; touches nothing external |
| `GET` | `/health/ready` | — | per-model registry versions and `ok` / `degraded` / `unavailable` (503 only when nothing resolves) |
| `GET` | `/metrics` | — | Prometheus scrape target |
| `POST` | `/events/new-data` | — | push trigger for the streaming pipeline (202) |
| `POST` | `/predict_pk` | `pk_model` | single primary key |
| `POST` | `/predict_cpk` | `composite_pk_model` | composite primary key |
| `POST` | `/predict_fk` | `fk_model` | single foreign key |
| `POST` | `/predict_cfk` | `composite_fk_model` | composite foreign key |
| `POST` | `/predict_normalform` | `denormalization_model` | normal form 0–3 |
| `POST` | `/predict_subject_area` | `subject_area_model` | subject-area name |

The request schema for each route is a Pydantic model in `webservice/data_model_*.py` —
that file is the source of truth for the exact feature list.

`/predict_subject_area` is the one endpoint whose request is text rather than column
statistics (`table_name` plus a comma-separated `columns` string), and whose
`prediction` is therefore a string rather than an int class. A table that fits no
discovered area still gets the nearest one, so a low `probability` — not a special
label — is the signal not to trust the answer.

---

## Data

Training data is metadata + statistics extracted from 60+ publicly available and
synthetically generated databases (Kaggle, TPC-H, Northwind, Spider, Sakila, Chinook,
MIMIC-IV, plus Python- and LLM-generated synthetic schemas). Only metadata is used, so
the datasets can be small and varied without any real records being stored.

| File | Rows (columns) | Databases | Tables | Used by |
| :--- | ---: | ---: | ---: | :--- |
| `data/summary_output_task_1_2_training.csv` | 10,348 | 241 | 1,706 | Tasks 1 & 2 |
| `data/nf_training.csv` | 6,575 | 1 | 432 | Task 3 |
| `data/nf_test_analyse.csv` | 13,822 | 503 | 1,560 | Task 3 (superseded) |
| `data/raw_metadata.csv` | 3,009 | 54 | 466 | raw extract / EDA |

`data/nf_training.csv` is not a hand-labelled extract like the others. It is *generated*:
[`task_3/nf_recipes.py`](task_3/nf_recipes.py) materialises 432 tables in Iceberg whose
normal form is known by construction, and [`nf_features.py`](prefect/nf_features.py)
profiles them through the same code path that serves live predictions. A build ends with
seven checks — among them that all 432 labels survive verification against the
dependencies actually discovered, and that no single feature predicts the label better
than a fixed threshold, which would mean the generator left a fingerprint.

### Class balance

Keys are rare by construction (a table has one PK and many ordinary columns), so the
targets are imbalanced — **accuracy is meaningless here; we track F1 and PR-AUC instead.**

| Task 1 / 2 target | Positive rate | | Task 3 class | Tables | Rows |
| :--- | ---: | --- | :--- | ---: | ---: |
| Single primary key | 12.0 % | | 0NF (violates 1NF) | 22.9 % | 20.1 % |
| Composite primary key | 8.7 % | | 1NF | 22.0 % | 24.2 % |
| Single foreign key | 16.5 % | | 2NF | 26.6 % | 28.5 % |
| Composite foreign key | 2.1 % | | 3NF | 28.5 % | 27.2 % |

Task 3 data is generated to keep the four classes balanced — a luxury the key tasks do not
have. Both units are shown because they differ: the label is per table, but the model
predicts per column, so a wide table contributes more rows than a narrow one. The
**table** column is the honest balance.

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

Features are engineered per column, per table, and — for the single-FK model — across
tables. Generated directly from the training scripts:

- **Tasks 1 & 2 — 31 features** (identical set; only the target differs), engineered in the
  training scripts from the bundled metadata CSV.
- **Task 2 single FK — 37 features**: 30 of the shared set plus **7 cross-table features**
  computed by [`src/cross_table_features.py`](src/cross_table_features.py). A foreign key
  is a property of a *pair* of columns in two tables, so this is the only model that looks
  outside the table being classified. They raised its cross-validated F1 from 0.696 to
  0.810; that module's docstring carries the measurements, including three further ideas
  that were measured and rejected.
- **Task 3 — 53 features**, computed by [`prefect/nf_features.py`](prefect/nf_features.py)
  by *measuring* each table in Trino. Nothing in this set is read from a CSV column
  someone typed.

> **Reference scope.** The cross-table features are relative to the set of other tables
> they may look at. Training scopes that to one `database`; the serving pipeline can only
> scope to a Trino schema, which may hold many unrelated tables. A model trained only on
> the narrow scope drops to F1 0.666 on a wide one — below the 0.696 it gets with no
> cross-table features at all. The fk model is therefore trained on several scope widths at
> once, which holds it at 0.783 even in the worst case. `cv_f1_by_scope_*` in MLflow reports
> the estimate per width.

Identifier columns (`database`, `schema`, `table_name`, `column_name`) and raw
`min_value` / `max_value` are never used as features — only for grouping and traceability.
The cross-table features are *derived from* the identifiers rather than using them
directly, which is why they are computed at runtime instead of stored in the CSV.

### Task 3: feature engineering

None of the 47 features are hand-labelled alongside the target — each is derived from
data the profiler discovers itself, which rules out a model partly reading the answer off
its own input:

- **Atomicity** is measured — separator density, token counts, the share of values that
  look like a packed list — instead of asserted by a `is_this_col_violating_1nf` flag.
- **Keys** come from a level-wise unique-column-combination search up to size 3, ranked by
  a birthday estimate. Nobody declares the primary key.
- **Dependencies** come from `count(DISTINCT …)` comparisons over column pairs, then get
  classified as partial or transitive by where the determinant sits relative to the
  discovered key.

`nf_features.py` is the single source of that contract: the training script's `X`, the
Pydantic request schema, the Prefect pipeline payload and the MLflow signature all read
`FEATURE_COLUMNS` from it, and a test fails if any of them drift apart. The four
`column_type_*` dummies are appended by the one-hot encoder — `bigint` is the reference
category and has no column of its own.

**Four feature groups are unusual enough to explain.**

*Cost budgets are features.* A profiler that must answer within a budget is guaranteed to
give partial answers on large tables. Rather than hide that, the budgets report themselves:
`table_sampled`, `table_sample_ratio`, `table_search_truncated`, `table_pair_coverage` and
`table_ucc_search_coverage` tell the model how completely the table was actually examined,
so it can discount a `table_fd_count` of 0 that means "none found" differently from one
that means "we stopped looking".

*Both a strict and a relaxed dependency count.* The textbook definition of a partial
dependency also requires the *dependent* to be non-prime. That is correct when the keys are
known and useless when they are discovered: a near-unique column forms a valid UCC with
almost every other column, so the union of all UCCs covers the table and everything comes
out prime. Measured on the generated set, the strict filter drove both counts to zero for
every single table. Both variants are therefore reported and the model decides which
carries signal.

*Approximate dependencies.* `table_near_fd_*` and `table_near_ucc_count` hold at a strength
threshold rather than exactly, which is what survives sampling and dirty data.
`table_has_near_key_but_no_key` is the specific "there is almost a key here" signal.

*Any-key judgement and pair-determinant search.* Added 2026-08-07 after a real-world run
(Willibald period-1) mispredicted 6 of 10 tables. The original partial/transitive counts
judge a dependency against the top-ranked key only; `table_partial_fd_count_anykey` and
`table_transitive_fd_count_anykey` judge the same dependencies against every discovered
key, which catches a determinant that only a lower-ranked key makes prime. The original
search also only ever compared single columns; `table_pair_fd_count` and its
`partial`/`transitive` split extend it to two-column determinants (`{a, b} -> c`), which
is how a rule like `{category_id, weight_class} -> shipping_tier` becomes visible at all.
Pairs are the more suggestible half of that search — almost any two columns "explain" a
third on a small table — so they are gated by an absolute row-support floor rather than a
ratio, calibrated against a real accidental pair sitting well below a real one.

> **What is still a limitation.** 3NF and 0NF tables have near-identical medians on every
> dependency feature, so the whole 0NF/3NF boundary rests on the atomicity detector. Its
> blind spot is a repeating group with obfuscated column names — two tables in the
> generated set are labelled 0NF with no violation any feature can see. See
> [TASK_3_PLAN.md](TASK_3_PLAN.md).
>
> The structural-key filter (`ucc_score >= 1`, see "Labelling conventions" below) has a
> second blind spot at the opposite end: on a table with only a handful of rows, a fully
> unique single column clears the bar for free (`ucc_score` is `rows / 2` there), whether
> its uniqueness means anything or not. Measured on Chinook's real `employee` table (8
> rows, never part of training): six ordinary columns each pass as a "structural" key on
> that basis alone, which turns `{phone} -> city` into a "partial dependency" that no
> designer intended — a coincidence of 8 people's data, not a rule. `LOW_EVIDENCE_MIN_ROWS`
> flags this table for review regardless (see below), so it does not reach a human
> unchecked, but the filter itself carries no row-count floor yet.

<details>
<summary><b>Task 3 feature dictionary — all 53</b> (click to expand)</summary>

Source of truth: `FEATURE_COLUMNS` and `COLUMN_TYPE_DUMMIES` in
[`prefect/nf_features.py`](prefect/nf_features.py). Listed in the order the model sees
them. `database`, `schema`, `table_name`, `column_name` and `column_type` are identity
columns and never features.

**Per column — shape and name (8)**

| Feature | Description | Type |
| :--- | :--- | :--- |
| `ordinal_position` | Position of the column in the table (1-based) | int |
| `col_unique_ratio` | Distinct values ÷ rows | float |
| `col_null_ratio` | NULLs ÷ rows | float |
| `col_list_like_ratio` | Share of values that look like a packed list | float |
| `col_mean_token_length` | Mean length of the tokens a value splits into | float |
| `col_mean_token_count` | Mean number of tokens per value | float |
| `col_separator_density` | Separator characters per character | float |
| `col_is_freetext_name` | The column's *name* suggests prose (`comment`, `description`, …) | 0/1 |

**Per column — position in the discovered structure (6)**

| Feature | Description | Type |
| :--- | :--- | :--- |
| `col_violates_1nf` | Measured non-atomic: list-like beyond threshold and not prose | 0/1 |
| `col_in_repeating_group` | Name belongs to a numbered family (`phone1`, `phone2`, …) | 0/1 |
| `col_in_candidate_key` | Part of the chosen minimal key | 0/1 |
| `col_is_prime` | Prime attribute (part of the primary key) | 0/1 |
| `col_determines_count` | Discovered dependencies where this column is the determinant | int |
| `col_depends_on_count` | Discovered dependencies where this column is the dependent | int |

**Per table — how completely it was examined (6)**

These describe the *measurement*, not the table. See "Cost budgets are features" above.

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_row_count` | Rows in the table | int |
| `table_sampled` | Profiling used a deterministic row sample | 0/1 |
| `table_sample_ratio` | Sampled rows ÷ total rows (1.0 when unsampled) | float |
| `table_search_truncated` | A budget cut the search short | 0/1 |
| `table_pair_coverage` | Share of column pairs actually tested for dependencies | float |
| `table_ucc_search_coverage` | Share of key candidates actually tested | float |

**Per table — atomicity (5)**

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_column_count` | Columns in the table | int |
| `table_max_list_like_ratio` | Highest `col_list_like_ratio` in the table | float |
| `table_ratio_list_like_columns` | Share of columns that look list-like | float |
| `table_repeating_group_ratio` | Share of columns in a numbered family | float |
| `table_constant_column_ratio` | Share of columns with exactly one distinct value | float |

**Per table — keys (5)**

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_has_no_ucc_le3` | No unique column combination found up to size 3 | 0/1 |
| `table_candidate_key_count` | Unique column combinations found | int |
| `table_key_size` | Columns in the chosen primary key | int |
| `table_composite_key_count` | Candidate keys with more than one column | int |
| `table_prime_ratio` | Prime attributes ÷ columns | float |

**Per table — functional dependencies (8)**

`partial` = determinant is a proper part of a composite key (breaks 2NF) ·
`transitive` = determinant sits outside every key (breaks 3NF) · `strict` = additionally
requires a non-prime dependent, the textbook definition.

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_fd_count` | Dependencies discovered | int |
| `table_fd_ratio` | …÷ the number of pairs tested | float |
| `table_partial_fd_count` | Of those, partial | int |
| `table_partial_fd_ratio` | …as a share | float |
| `table_transitive_fd_count` | Of those, transitive | int |
| `table_transitive_fd_ratio` | …as a share | float |
| `table_strict_partial_fd_count` | Partial under the strict definition | int |
| `table_strict_transitive_fd_count` | Transitive under the strict definition | int |

**Per table — any-key and pair-determinant dependencies (6)**

Added 2026-08-07 after the Willibald period-1 run exposed two blind spots: the
partial/transitive counts above are classified against the top-ranked key only, and the
dependency search only ever looked at single-column determinants. `_anykey` judges the
same discovered dependencies against *every* candidate key instead of just the top-ranked
one; `table_pair_*` extends the search itself to two-column determinants
(`{a, b} -> c`), gated by an absolute support floor (`MIN_PAIR_FD_SUPPORT`) so a pair that
repeats in too few rows to be more than coincidence is not counted.

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_partial_fd_count_anykey` | Partial dependencies, judged against all discovered keys | int |
| `table_transitive_fd_count_anykey` | Transitive dependencies, judged against all discovered keys | int |
| `table_pair_fd_count` | Two-column-determinant dependencies discovered | int |
| `table_pair_partial_fd_count` | Of those, partial | int |
| `table_pair_transitive_fd_count` | Of those, transitive | int |
| `table_pair_fd_search_coverage` | Share of eligible column pairs actually tested | float |

**Per table — approximate structure (5)**

| Feature | Description | Type |
| :--- | :--- | :--- |
| `table_near_fd_count` | Dependencies holding for ≥ 95 % of rows | int |
| `table_near_fd_ratio` | …as a share of pairs tested | float |
| `table_max_near_fd_strength` | Strength of the strongest near-dependency | float |
| `table_near_ucc_count` | Column combinations that are nearly unique | int |
| `table_has_near_key_but_no_key` | A near-key exists but no exact key was found | 0/1 |

**Data type (4)**

One-hot over the five types that occur in the data. `bigint` is the reference category and
therefore has no column — it is 89 % of all rows, so encoding it too would be collinear.

| Feature | Type |
| :--- | :--- |
| `column_type_date`, `column_type_double`, `column_type_integer`, `column_type_varchar` | bool |

</details>

<details>
<summary><b>Tasks 1 &amp; 2 metadata &amp; statistics dictionary</b> (click to expand)</summary>

`T1` = single/composite primary key · `T2` = single/composite foreign key ·
`identifier` = used for grouping only · `—` = present in the data but not used as a
training feature.

| Variable | Description | Example | Used in |
| :--- | :--- | :--- | :--- |
| `database` | Name of the database the table lives in | "Formula 1" | identifier |
| `schema` | Name of the schema the table lives in | "Season 2025" | identifier |
| `table_name` | Name of the table | "Drivers" | identifier |
| `column_name` | Name of the column | "Age" | identifier |
| `column_type` | Raw data type (one-hot encoded for training) | "int" | — |
| `min_value` | Minimum value of the column | "18" | — |
| `max_value` | Maximum value of the column | "44" | — |
| `number_unique_values` | Count of distinct values in the column | "23" | T1, T2 |
| `count` | Number of rows in the table | "25" | T1, T2 |
| `null_count` | Raw count of null values in the column | "0" | — (dropped, redundant) |
| `null_ratio` | `null_count / count` | "0.0" | — (dropped, redundant) |
| `is_unique` | Column values are fully unique | 0 or 1 | T1, T2 |
| `ordinal_position` | Position of the column in the table (1-based) | "2" | T1, T2 |
| `unique_ratio` | `number_unique_values / count` | "0.177" | T1, T2 |
| `is_non_null` | Column has no null values | 0 or 1 | — (dropped, redundant) |
| `is_first_column` | Column is the first in the table | 0 or 1 | T1, T2 |
| `relative_ordinal_position` | `ordinal_position / table_column_count` | "0.222" | T1, T2 |
| `is_first_unique_column` | Column is the first unique column in the table | 0 or 1 | T1, T2 |
| `table_column_count` | Total columns in the table | "9" | T1, T2 |
| `table_unique_column_count` | Number of fully unique columns in the table | "0" | T1, T2 |
| `table_row_count` | Number of rows in the table | "768" | T1, T2 |
| `other_unique_columns_in_table` | Count of *other* unique columns in the table | "0" | — (dropped, redundant) |
| `table_has_unique_column` | Table has at least one unique column | 0 or 1 | T1, T2 |
| `table_has_no_single_pk_candidate` | No single-column PK candidate exists | 0 or 1 | T1, T2 |
| `table_near_unique_column_count` | Number of near-unique columns in the table | "0" | T1, T2 |
| `table_id_named_column_count` | Number of ID-named columns in the table | "0" | T1, T2 |
| `table_non_null_column_count` | Number of non-null columns in the table | "9" | T1, T2 |
| `table_max_unique_ratio` | Highest `unique_ratio` in the table | "0.671" | T1, T2 |
| `unique_ratio_rank` | Rank of this column's `unique_ratio` in the table | "4" | T1, T2 |
| `null_ratio_rank` | Rank of this column's `null_ratio` in the table | "2" | T1, T2 |
| `is_least_null_in_table` | Column has the lowest `null_ratio` in the table | 0 or 1 | T1, T2 |
| `unique_ratio_relative_to_max` | `unique_ratio / table_max_unique_ratio` | "0.264" | T1, T2 |
| `other_near_unique_columns_in_table` | Count of *other* near-unique columns | "0" | — (dropped, redundant) |
| `name_ends_with_id` | Column name ends with "id" | 0 or 1 | T1, T2 |
| `name_contains_key` | Column name contains "key" | 0 or 1 | — (dropped, redundant) |
| `name_contains_table_name` | Column name contains the table name | 0 or 1 | T1, T2 |
| `name_is_singular_table_id` | Column name = singular table name + "id" | 0 or 1 | T1, T2 |
| `name_length` | Length of the column name | "7" | T1, T2 |
| `column_type_boolean` | One-hot: boolean | 0 or 1 | T1, T2 |
| `column_type_date` | One-hot: date | 0 or 1 | T1, T2 |
| `column_type_decimal` | One-hot: decimal | 0 or 1 | T1, T2 |
| `column_type_double` | One-hot: double | 0 or 1 | T1, T2 |
| `column_type_integer` | One-hot: integer | 0 or 1 | T1, T2 |
| `column_type_varchar` | One-hot: varchar | 0 or 1 | T1, T2 |
| `table_contains_1nf_violation` | Table has a 1NF violation | 0 or 1 | — (dropped, leakage guard) |
| `pk_target` | Is the column a single primary key? | 0 or 1 | **target: T1** |
| `composite_pk_target` | Is the column part of a composite PK? | 0 or 1 | **target: T1** |
| `fk_target` | Is the column a foreign key? | 0 or 1 | **target: T2** |
| `composite_fk_target` | Is the column part of a composite FK? | 0 or 1 | **target: T2** |

</details>

### Data types

The current pipeline covers standard scalar types. Future work should add types like
`blob`, while intentionally leaving out semi-structured data such as `json` and `xml`.

---

## Model & results

Task 1 and Task 2 both use a **RandomForestClassifier** with `class_weight="balanced"` and otherwise
default hyperparameters. The train/test split is **grouped by database** (`StratifiedGroupKFold`).

Example F1 scores from the registry:

| Model | Train F1 | Test F1 |
| :--- | ---: | ---: |
| `composite_pk_model` | 0.978 | 0.712 |
| `composite_fk_model` | 1.000 | 0.811 |

The high train F1 is an un-tuned RandomForest memorising the training set. Tuning it is a
non-goal — the graded surface is the engineering. These two models have no CI quality gate
yet; Task 3 below has one, and it is the pattern the key models should follow.

### Task 3

Four candidates are cross-validated and the best is registered: XGBoost and LightGBM, each
plain and with a randomised search. The reported number is the mean over **five**
`StratifiedGroupKFold` folds, not one split — with ~85 tables per holdout, a single draw
swings several points.

The grouping is by **recipe and matched pair**, joined with union-find, not by
`table_name`: the generated set contains matched pairs — the same schema once with a 0NF
violation injected and once without — that carry different recipe ids and opposite
labels. Grouping by table name alone would let a model see one half of a pair in training
and be scored on the other. Linking recipe and pair leaves 327 groups and zero pairs
spanning the split.

| | Column F1 | Table accuracy |
| :--- | ---: | ---: |
| `denormalization_model` (XGBoost, randomised search) | 0.9823 ± 0.0182 | **0.9724 ± 0.0261** |

Table accuracy is the headline, because the pipeline stores one normal form per table: in
the column-level metric a 40-column table would otherwise count eight times as much as a
five-column one.

[`test/test_models/test_normalform_baseline.py`](test/test_models/test_normalform_baseline.py)
holds this to a floor in CI. CI has neither MLflow nor Trino and so cannot retrain, but
every training run writes `data/nf_baseline.json`, and a worse run fails the gate. The
floors are the measured mean minus two standard deviations — the only way past them is
lowering a number in a visible diff.

#### Serving-time cross-check and the Willibald eval gate

Every prediction [`normalform_pipeline.py`](prefect/normalform_pipeline.py) stores in
`iceberg.prediction_results.nf_results` carries a second, independently-computed reading
of the same evidence next to the model's class — never as a feature, only as a check:

| Column | What it is |
| :--- | :--- |
| `rule_normal_form` | The class a deterministic decision procedure (`nf_features.derive_rule_normal_form`) reads off the discovered structure — keys, dependencies, atomicity violations — from the same profiling pass that built the features. No learning involved. |
| `rule_agrees` | Whether `rule_normal_form` matches the model's prediction. |
| `needs_review` | `True` when the two disagree, or when the table has fewer than 30 rows (`LOW_EVIDENCE_MIN_ROWS`) — too few for any dependency statistic to mean more than coincidence. |
| `review_reasons` | Plain text saying which of the above triggered the flag. |
| `constant_columns` | Informational only, never a review trigger — a constant column can hide a real rule behind it (`Ort -> Land` with only one country loaded) that only becomes visible once the data varies. |

This exists because confidence turned out not to be a usable review signal: on a real-world
run (Willibald period-1) every misprediction sat at 0.97+ confidence, indistinguishable
from the correct ones — the model is a deterministic function of features that had simply
missed the decisive dependency. A second, rule-based reading of the same evidence catches
what confidence structurally cannot: disagreement between two independent judgements of the
same data.

**The Willibald eval gate.** [`task_3/nf_willibald_eval.py`](task_3/nf_willibald_eval.py)
snapshots features for the 10 real Willibald DWA-Challenge tables (see Source databases
above) and every training run scores its best candidate against them, logging
`willibald_table_accuracy` to MLflow and into `data/nf_baseline.json`. It never trains
anything — unlike the generated set, this is real data nobody constructed to fit the
model's assumptions, so it stays a held-out gate, floored at 0.8 by
[`test/test_models/test_normalform_baseline.py`](test/test_models/test_normalform_baseline.py).
The ground truth it is scored against
([`task_3/nf_ground_truth_willibald.py`](task_3/nf_ground_truth_willibald.py) →
`data/willibald_ground_truth.csv`) runs the exact same `derive_rule_normal_form`
decision procedure over the raw seed data, exhaustively rather than under a search budget
— model and ground truth are read with one ruler, not two.

---

## MLOps stack

| Concern | Tool | Status |
| :--- | :--- | :--- |
| Experiment tracking & registry | MLflow | ✅ params, metrics, signature, feature list; alias-based deploy |
| Model service | FastAPI + Docker | ✅ 6 typed endpoints, `/health/live` + `/health/ready`, `/metrics` |
| CI | GitHub Actions | ✅ Ruff lint/format + pytest with coverage |
| Vulnerability scanning | Trivy | ✅ `scripts/scan_vulnerabilities.sh` scans every compose image, renders a PDF report under `security-reports/` |
| Service monitoring | Prometheus + Grafana | ✅ golden signals, 10 alert rules, 5 provisioned dashboards |
| Model monitoring | Evidently | ✅ input drift against a real reference set |
| Data pipeline | Prefect | ✅ event-driven — a webhook (`POST /events/new-data`) or a cron safety net triggers watermark-based change detection, which fans out to three parallel prediction pipelines (keys, normal form, subject area) over the changed tables, plus a separately cron-scheduled quality backtest against the labelled holdout |
| Retraining | `scripts/promote_model.py` | ✅ metric-gated `dev`→`prod` alias promotion (`test_f1_score` by default; `--model` / `--metric` / `--dry-run`). Models are cached per name via `@lru_cache`, so a promotion still needs `docker compose restart model-service` to take effect ([step 2c](#2c-restart-the-model-service-after-promoting)) |

### Monitoring detail

- `webservice/metrics.py` adds model-level metrics on top of the HTTP golden signals:
  predictions by class, inference duration, errors by type, returned confidence.
- `prometheus/alert.yaml` holds 10 rules in three groups: `service-health` (instance down,
  model service unreachable), `golden-signals` (5xx/4xx rate, p95 latency, no traffic,
  memory) and `model-health` (a model stuck on one class, confidence collapse, prediction
  errors spiking).
- `grafana/dashboards/` holds five auto-provisioned boards — `model_service_golden_signals`
  plus a drift and a quality board each for the key and the normal-form models. A fresh
  `docker compose up` shows all of them with no manual setup.
- `evidently_service/build_monitoring_references.py` regenerates the drift reference sets
  from the training data. **Re-run it whenever the feature set changes**, then rebuild the
  image so the new baseline is baked in — see [step 2c](#2c-restart-the-model-service-after-promoting).
- **Two different F1s are on the quality board, on purpose.** The Evidently
  `evidently_clf_f1score` series comes from the Prefect backtest, which replays the
  labelled holdout through the live model — and since the key models are refit on every
  labelled row, those rows were in its training set. It is a *regression canary*: a drop
  means something broke, but the level says nothing about unseen data. The honest number is
  `model_offline_f1`, the cross-validated score of the version actually being served, which
  the model service reads off its MLflow run at startup. The `estimator` label separates
  `cv_mean_5fold_grouped` (fk, and cpk once retrained) from the older
  `single_fold_holdout`, whose fold-to-fold spread was measured at up to 0.21 F1 — so a
  model still on the old estimator cannot silently read as if it were cross-validated.
  `model_served_version` sits alongside it, so a quality change can be lined up against a
  deployment.

---

## Testing

```bash
pytest                              # unit + API tests, with coverage
pytest --cov-report=term-missing    # see uncovered lines
ruff check . && ruff format --check .
```

`test/test_api/` covers the health probes against an in-process app, including the
degraded and unavailable states. `test/test_webservice/` covers the six predict routes
(success shape, feature forwarding, validation errors, model-failure mapping, monitoring
events), `/metrics`, and the `/events/new-data` webhook. `test/test_models/` checks the
model/schema contract between the Pydantic request models and the registered MLflow
signatures, skipping when no stack is reachable. `test/test_data/` validates the training
CSVs; `test/test_task_3/` validates the synthetic normal-form generator (recipes,
labelling, manifest, the split used for cross-validation). `test/test_pipelines/` covers
the Prefect change-detection and prediction flows against fakes, with no live Trino
needed. `test/test_features/` covers the cross-table feature module, and
`test/test_scripts/` covers the promotion gate.

---

## Project structure

```
webservice/                     FastAPI app, Pydantic schemas, predict + metrics
task_1/ task_2/ task_3/ task_4/ training + registration scripts, one set per model
prefect/                        change detection + prediction pipelines, quality backtest
trino-iceberg/                  Trino/Nessie config, dev-credential generator
dbt/                            dbt project scaffold (Trino profile)
evidently_service/              drift-monitoring service + reference builder
prometheus/ alertmanager/       scrape config + alert rules + routing
grafana/                        provisioned datasource + dashboards
test/                           API, pipeline, prediction and data-quality tests
scripts/                        stack setup, promotion, vulnerability scan, DooD launcher
curl_tests/                     one smoke-test script per predict route
data/                           training CSVs
documentation/                  MLOps plan, presentations
```

---

## Tech stack

Python · scikit-learn · XGBoost · LightGBM · sentence-transformers / UMAP / HDBSCAN ·
Ollama · FastAPI · Pydantic · MLflow · Prefect · Docker Compose · Trino · Nessie · MinIO ·
Prometheus · Grafana · Evidently · Trivy · Ruff · pytest · GitHub Actions · uv

---

## Contributors

Christian · Niklas · Nijat
