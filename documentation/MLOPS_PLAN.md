# MLOps Integration Plan - Alligator Metaswamp

## Übersicht
Dieses Dokument beschreibt die vollständige MLOps-Integration für das PK/FK-Detection-Projekt mit strukturiertem Code, CI/CD, Data Pipelines, Model Tracking und Monitoring.

---

## 1. WORKFLOW-DIAGRAMM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTWICKLUNGSPHASE                                │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   Code Development          │
                    │   - Ruff Linting            │
                    │   - Type Hints              │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Local Testing             │
                    │   - pytest Unit Tests       │
                    │   - Integration Tests       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Git Push to GitHub        │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                         CI/CD PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   GitHub Actions Trigger    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Lint & Test Job           │
                    │   - Ruff Check              │
                    │   - pytest --cov            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Docker Build Job          │
                    │   - Build Images            │
                    │   - Push to GHCR            │
                    │   - Tag: latest, SHA        │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE (Prefect + dbt)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Test Data Generation      │
                    │   - dbt Seeds/Models        │
                    │   - Synthetic Data          │
                    └──────────────┬──────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                              │
┌───────────▼──────────┐                      ┌───────────▼──────────┐
│  BATCH Processing    │                      │  STREAMING Events    │
│  - CSV Ingestion     │                      │  - Message Queue     │
│  - Batch Training    │                      │  - Real-time Predict │
│  - Batch Prediction  │                      │  - OpenAPI Calls     │
└───────────┬──────────┘                      └───────────┬──────────┘
            │                                              │
            └──────────────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Prefect Orchestration     │
                    │   - Task Scheduling         │
                    │   - Flow Management         │
                    │   - Error Handling          │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE (MLflow)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Model Training            │
                    │   - Log Parameters          │
                    │   - Log Metrics             │
                    │   - Log Artifacts           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   MLflow Model Registry     │
                    │   - Register Model          │
                    │   - Version Control         │
                    │   - Stage: Staging/Prod     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Model Validation          │
                    │   - Performance Checks      │
                    │   - A/B Testing Ready       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Promotion to Production   │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   FastAPI Service           │
                    │   - Load Model from MLflow  │
                    │   - Typed Endpoints         │
                    │   - Pydantic Validation     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Docker Container          │
                    │   - Model Service           │
                    │   - Health Checks           │
                    │   - Metrics Endpoint        │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & OBSERVABILITY                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                              │
┌───────────▼──────────┐                      ┌───────────▼──────────┐
│  Service Monitoring  │                      │  Model Monitoring    │
│  - Prometheus        │                      │  - Evidently         │
│  - Golden Signals:   │                      │  - Input Drift       │
│    * Latency         │                      │  - Prediction Qual.  │
│    * Traffic         │                      │  - Data Quality      │
│    * Errors          │                      └───────────┬──────────┘
│    * Saturation      │                                  │
└───────────┬──────────┘                                  │
            │                                              │
            └──────────────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Grafana Dashboards        │
                    │   - Real-time Metrics       │
                    │   - Alerts                  │
                    │   - Historical Analysis     │
                    └─────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Alert Notifications       │
                    │   - Slack/Email/PagerDuty   │
                    │   - Threshold Violations    │
                    └─────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Retraining Trigger         │
                    │   (Manual or Automated)      │
                    └──────────────────────────────┘
```

---

## 2. DETAILLIERTE TODO-LISTE

> **Stand: 27. Juli 2026** — abgeglichen mit dem tatsächlichen Repo-Zustand (Branch `dev`).
> Legende: `[x]` erledigt · `[~]` teilweise erledigt (Details im Unterpunkt) · `[ ]` offen · ~~durchgestrichen~~ = bewusst verworfen.

**Fortschritt auf einen Blick**

| Phase | Status | Kern-Lücke |
|-------|--------|------------|
| 1 Clean Code | 🟢 ~90% | Pre-commit Hooks fehlen |
| 2 Testing & CI/CD | 🟡 ~45% | Keine Unit-Tests für API/Predict, keine Coverage, kein Docker-Build-Workflow |
| 3 API & Docker | 🟡 ~55% | Kein API-Versioning, Dockerfile nicht gehärtet (root, single-stage) |
| 4 Data Pipeline | 🟢 ~80% | Streaming steht; Training nicht orchestriert, keine Deployments für Batch |
| 5 MLflow & Retraining | 🟡 ~70% | `dev`→`prod` Promotion mit F1-Gate steht; Retraining selbst (Trigger, Prefect-Flow) offen |
| 6 Monitoring | 🟡 ~55% | Grafana-Dashboards leer, Evidently noch auf Tutorial-Daten |

---

### Phase 1: Clean, Structured Code

- [~] **1.1** Ruff Installation und Konfiguration
  - [x] Ruff zu `pyproject.toml` hinzufügen (gepinnt auf `0.16.0`)
  - [x] Ruff-Konfiguration erstellen → eigene [`ruff.toml`](../ruff.toml) (line-length, rules, excludes inkl. `README.md`)
  - [ ] Pre-commit Hooks einrichten → **`.pre-commit-config.yaml` existiert nicht**; Ruff läuft aktuell nur in CI und manuell

- [x] **1.2** Code-Quality-Verbesserungen
  - [x] Bestehenden Code mit Ruff formatiert (`ruff check` + `ruff format` grün in CI)
  - [x] Type Hints in Training-Scripts, Prefect-Flows und Webservice

- [x] **1.3** Projektstruktur-Refactoring
  - [x] Klare Module-Trennung: [task_1/](../task_1/) (PK/CPK), [task_2/](../task_2/) (FK/CFK), [task_3/](../task_3/) (Normalform), [prefect/](../prefect/), [webservice/](../webservice/), [dbt/](../dbt/), [test/](../test/)
  - [x] Pipeline-Dateien umbenannt (`pk_fk_pipeline.py`, `normalform_pipeline.py`)
  - [ ] Aufräumen: `src/test_trino_connection.py` liegt außerhalb der Teststruktur; `mlflow.db`/`mlruns/` sind lokale Altlasten neben dem Postgres-Backend

---

### Phase 2: Testing & CI/CD

- [~] **2.1** pytest Setup
  - [~] `pytest` ist Hauptabhängigkeit; **`pytest-cov` und `pytest-mock` fehlen**
  - [x] Test-Verzeichnisstruktur [test/test_data/](../test/test_data/)
  - [x] [conftest.py](../test/test_data/conftest.py) für Fixtures
  - [ ] pytest-Konfiguration in `pyproject.toml` (`[tool.pytest.ini_options]`, testpaths, addopts) fehlt
  - [ ] `[project.optional-dependencies] dev` fehlt — CI ruft `uv pip install -e ".[dev]"` auf ein nicht existierendes Extra auf

- [~] **2.2** Unit Tests schreiben
  - [x] Datenqualitäts-Tests für Trainingsdaten: [test_training_data_quality.py](../test/test_data/test_training_data_quality.py), [..._nf.py](../test/test_data/test_training_data_quality_nf.py), [..._subject_area.py](../test/test_data/test_training_data_quality_subject_area.py)
  - [ ] Tests für Prediction-Logik ([webservice/predict.py](../webservice/predict.py), MLflow-Model gemockt)
  - [ ] Tests für FastAPI Endpoints mit `TestClient` (alle 5 Predict-Routes + Fehlerpfad 400)
  - [ ] Tests für Pydantic-Validierung (`data_model_*.py`)
  - [ ] Tests für Prefect-Tasks (Feature-Berechnung, `recompute_table_stats`, Queue-Logik)
  - [ ] Mindestens 70% Code Coverage (aktuell wird Coverage gar nicht gemessen)

- [~] **2.3** GitHub Actions CI Pipeline
  - [x] [.github/workflows/ci.yml](../.github/workflows/ci.yml) mit Lint-, Test- und Summary-Job
  - [x] Lint Job (`ruff check` + `ruff format --check`)
  - [~] Test Job läuft `pytest` — **ohne `--cov`**, die Codecov-/HTML-Upload-Steps laden daher nie existierende Artefakte hoch
  - [~] Matrix Testing vorhanden, aber nur `['3.11']` → um 3.12 erweitern oder Matrix entfernen
  - [ ] Coverage Badge zu README hinzufügen
  - [ ] Trigger prüfen: Workflow horcht auf `develop`, der aktive Branch heißt `dev` → Pushes lösen aktuell keine CI aus

- [ ] **2.4** Docker Build & GHCR Pipeline  *(offen — TODO „Docker build über github actions")*
  - [ ] `.github/workflows/docker-build.yml` erstellen
  - [ ] Multi-stage Dockerfile optimieren (siehe 3.3)
  - [ ] Docker Build und Push zu GHCR
  - [ ] Image Tagging (latest, SHA, semantic versioning)
  - [ ] GitHub Secrets für GHCR_TOKEN einrichten
  - [ ] `docker-compose.yaml` auf GHCR-Images umstellen (baut heute lokal aus `webservice/` und `evidently_service/`)

---

### Phase 3: Model Service API & Docker

- [~] **3.1** FastAPI Service Verbesserungen
  - [x] Pydantic Models für **alle 5** Endpoints: [data_model_pk.py](../webservice/data_model_pk.py), `_cpk`, `_fk`, `_cfk`, [data_model_denormalization.py](../webservice/data_model_denormalization.py)
  - [x] Endpoints `/predict_pk`, `/predict_cpk`, `/predict_fk`, `/predict_cfk`, `/predict_normalform` inkl. Probability im Response
  - [x] Error Handling mit `HTTPException` (400 bei Input-/Predict-Fehlern) in [app.py](../webservice/app.py)
  - [x] Smoke-Tests per curl: [curl_tests/](../curl_tests/)
  - [ ] API Versioning (`/api/v1/…`) — Doku [STREAMING_PREDICTION_APPROACHES.md](STREAMING_PREDICTION_APPROACHES.md) beschreibt bereits v1-Pfade, Code nutzt sie nicht
  - [ ] Health-Check erweitern: `/health/live` + `/health/ready` (heute nur `GET /` mit Statischer Message)
  - [ ] `/model/info` Endpoint (Modellname, Version, Alias)
  - [ ] OpenAPI-Schema anreichern (Beschreibungen, `examples`, Tags)
  - [ ] Auskommentierten Evidently-Block in `app.py` entfernen oder reaktivieren

- [~] **3.2** Model Loading Optimization
  - [x] Model Caching über `@lru_cache(maxsize=5)` in [predict.py](../webservice/predict.py)
  - [x] **Bugfix 27.07.:** `predict_proba` lief am rohen sklearn-Modell mit unsortierten Spalten, während `model.predict()` über pyfunc umsortiert. Sobald die Pydantic-Feldreihenfolge von der Trainingsreihenfolge abwich (cpk, cfk), scheiterte jeder Call mit `ValueError: feature names should match`. `_align_to_signature()` richtet die Spalten jetzt einmal an der Modellsignatur aus und meldet Schema-Drift mit Feldnamen statt einer nichtssagenden sklearn-Meldung
  - [x] **Bugfix 27.07.:** `ForeignKey` in [data_model_fk.py](../webservice/data_model_fk.py) war nicht mit `fk_model` synchron (4 Felder fehlten, 5 überzählig). Ursache: das fk-Trainingsskript wählt Features per **Drop-Liste**, das Pydantic-Schema wurde bei einer Erweiterung nicht nachgezogen
  - [x] **Schema-Drift-Contract-Test:** [test/test_models/test_model_schema_contract.py](../test/test_models/test_model_schema_contract.py) vergleicht für alle 5 Modelle die MLflow-Signatur gegen die Pydantic-Felder — Menge **und** Reihenfolge, getrennt geprüft. Hat sofort die verbliebene Reihenfolgen-Drift in `CompositePrimaryKey` und `CompositeForeignKey` gefunden (beide korrigiert). Skippt sauber, wenn kein MLflow erreichbar ist
  - [ ] Contract-Test in CI aktivieren — läuft dort mangels MLflow-Stack aktuell nur als Skip; sinnvoll wäre ein Job, der den Compose-Stack hochfährt
  - [ ] Model Loading beim Startup (heute lazy beim ersten Request → erster Call ist langsam)
  - [ ] Graceful Model Reload ohne Downtime (Cache-Invalidierung / Reload-Endpoint)
  - [ ] Model Alias/Version als Env Variable — `alias = "dev"` ist hart kodiert
  - [ ] `load_dotenv()` aus dem Request-Pfad in den Startup ziehen

- [~] **3.3** Dockerfile Optimierung ([webservice/Dockerfile](../webservice/Dockerfile))
  - [x] Minimal Base Image (`python:3.11.13-slim-bookworm`, `libgomp1` für LightGBM)
  - [ ] Multi-stage Build (builder + runtime)
  - [ ] Layer Caching optimieren (`COPY requirements.txt` vor `COPY . /app`)
  - [ ] Security: Non-root User
  - [ ] `.dockerignore` erstellen (aktuell wandert u.a. `__pycache__` ins Image)
  - [ ] `HEALTHCHECK` im Dockerfile

---

### Phase 4: Data Pipeline

- [~] **4.1** dbt Setup
  - [x] `dbt-core` + **`dbt-trino`** installiert (statt dbt-duckdb: DuckDB wird als Trino-Katalog angesprochen)
  - [x] dbt Projekt initialisiert: [dbt/dbt_project.yml](../dbt/dbt_project.yml)
  - [x] [profiles.yml](../dbt/profiles.yml) für Trino/LDAP konfiguriert (TODO „change dbt profile" erledigt)
  - [x] Seeds für Test-Daten: [dbt/seeds/](../dbt/seeds/) (`employee_3.csv`, `employee_not_3.csv`)
  - [~] ~~Models für Data Transformation~~ — `dbt/models/` ist leer: die Feature-Extraktion wurde bewusst nach Prefect verschoben, weil dbt-trino keine Python-Models unterstützt (siehe Docstring in [pk_fk_pipeline.py](../prefect/pk_fk_pipeline.py))
  - [ ] Entscheidung dokumentieren: bleibt dbt nur für Seeds/Tests, oder SQL-Staging-Models nachziehen?
  - [ ] `dbt/logs/` und `dbt/target/` in `.gitignore` aufnehmen (liegen aktuell untracked im Repo)

- [~] **4.2** Test Data Generation
  - [x] dbt Seed-Dateien mit synthetischen Normalform-Daten
  - [x] Schema-/Datendokumentation: [test/test_data/README.md](../test/test_data/README.md), `README_NF.md`, `README_SubjectArea.md`
  - [x] Datenqualitätstests — als **pytest**-Suite umgesetzt (nicht als dbt tests)
  - [ ] `dbt/test/` ist leer (nur `.gitkeep`) → entweder dbt-Tests ergänzen oder Verzeichnis entfernen
  - [ ] Python-Script zur erweiterten Datengenerierung

- [~] **4.3** Prefect Orchestration Setup
  - [x] Prefect 3.7 installiert (`prefect`, `prefect-sqlalchemy`, `prefect-dbt[trino]`)
  - [x] Prefect Server als Service in [docker-compose.yaml](../docker-compose.yaml) (Port 4200)
  - [x] Flow für Batch Prediction: `key-prediction-pipeline` in [pk_fk_pipeline.py](../prefect/pk_fk_pipeline.py)
  - [x] Flow für Normalform-Prediction: `normalform-prediction-pipeline` in [normalform_pipeline.py](../prefect/normalform_pipeline.py) (TODO „predict normalform into prefect flow" erledigt)
  - [ ] Flow für Batch **Training** — die fünf `task_*_train_and_register.py` laufen weiterhin standalone
  - [~] ~~Flow für dbt Run orchestrieren~~ — entfällt, solange keine dbt-Models existieren
  - [ ] Prefect Deployments + Scheduling (kein `serve()`/`deploy()`/Cron im Code)

- [~] **4.4** Batch Pipeline
  - [x] Feature-Extraktion aus `information_schema` → `staging.stg_column_features`
  - [x] Prediction-Queue + `fetch_new_rows` / `predict_batch` / `store_predictions_to_trino`
  - [x] Aggregation & Speicherung der Normalform-Ergebnisse in eigener Tabelle
  - [x] Error Handling und Retry Logic (`retries=2`, `retry_delay_seconds`)
  - [x] Keine „rows processed"-Meldung mehr bei leeren Predictions (TODO erledigt)
  - [x] Ergebnis-Export nach Trino; CSV-Beispiel: [data/summary_output_task_3_predict.csv](../data/summary_output_task_3_predict.csv)
  - [ ] Dedizierter CSV-Batch-Prediction-Flow inkl. Input-Validation
  - [ ] Training-Flow mit MLflow Logging (siehe 4.3)

- [~] **4.5** Streaming Pipeline — **umgesetzt am 27.07.2026**, siehe [STREAMING_PIPELINE.md](STREAMING_PIPELINE.md)
  - [x] Optionen dokumentiert: [STREAMING_PREDICTION_APPROACHES.md](STREAMING_PREDICTION_APPROACHES.md)
  - [x] Entscheidung: Push-Webhook **+** Watermark-Polling als Sicherheitsnetz, Prefect Events als Transport
  - [x] ~~Message Queue Setup (RabbitMQ/Kafka/Redis)~~ — verworfen: Prefect Events + `duckdb.staging.pending_changes` erfüllen die Aufgabe ohne neuen Service; echte CDC ist auf DuckDB-über-Trino ohnehin nicht möglich
  - [x] FastAPI Webhook: `POST /events/new-data` in [app.py](../webservice/app.py) — publiziert per REST, ohne Prefect-SDK im Model-Image
  - [x] Change Detector: [change_detector.py](../prefect/change_detector.py) — diffed `row_count`/`column_count` gegen `duckdb.staging.source_watermarks`
  - [x] Event-getriggerte Deployments inkl. Cron-Fallback (alle 15 min) und Concurrency-Limits: [serve_flows.py](../prefect/serve_flows.py)
  - [x] Re-Prediction geänderter Tabellen: Dedup-Sperre in beiden Pipelines aufgehoben, Ergebnisse werden historisiert (append)
  - [x] Recovery verwaister Claims nach hartem Prozessabbruch
  - [x] `prefect`-Service ins `ml-services-monitoring`-Netz geholt und um den `serve()`-Runner erweitert (Server + Flow-Runner in einem Container)
  - [x] Prefect-Serverstate in Postgres statt SQLite: eigene DB `prefect` neben `mlflow_db`, angelegt von [ensure_database.py](../prefect/ensure_database.py) — überlebt jetzt `docker compose down`
  - [x] `prefect`-DB ins stündliche S3-Backup aufgenommen: eigener Service `prefect_postgres_backup` (Prefix `prefect_db_backups`, 7 Tage Retention) — das Image sichert nur je eine DB pro Container
  - [ ] `prefect/Dockerfile` bauen, um das `pip install` beim Containerstart zu ersetzen
  - [ ] Event Logging zu Monitoring (Prometheus-Metriken für empfangene Events, Detection-Latenz, Backlog-Tiefe)
  - [ ] Housekeeping-Job: abgeschlossene `pending_changes`-Zeilen nach N Tagen löschen
  - [ ] Automatisierte Tests für Detector und Claim-Logik (bisher nur sqlglot-Parse + TestClient-Check)

---

### Phase 5: MLOps Tracking, Versioning, Retraining

- [~] **5.1** MLflow Tracking vollständig integrieren
  - [x] Experiment Tracking in **allen 5** Training-Scripts (`mlflow.start_run` je Kandidat)
  - [x] Hyperparameter-Suche mit `RandomizedSearchCV` + `StratifiedGroupKFold` in allen 5 Scripts (statt Optuna) — TODO „more than one model … with hyperparameter search" erledigt
  - [x] Metrics Logging (Accuracy, Precision, Recall, F1 — je train/test)
  - [x] Alle vier Kandidaten (RF, XGB, jeweils Baseline + RandomizedSearch) werden geloggt, bester per `f1_score` registriert
  - [x] MLflow-Backend produktionsnah: Postgres als Backend-Store, MinIO/S3 als Artifact-Store, stündliches Postgres-Backup nach S3
  - [~] Artifacts: Confusion Matrix als `mlflow.log_dict` ✔ — **Feature Importance Plot fehlt**
  - [ ] Dataset Tracking mit `mlflow.data` / `log_input`

- [~] **5.2** MLflow Model Registry
  - [x] Model Registration nach Training (`mlflow.register_model`)
  - [x] Model Versioning über die Registry
  - [x] Model Metadata (Tags via `mlflow.set_tags`)
  - [x] Model Signature (`infer_signature`) und Input Example
  - [x] **Aliase statt Stages**: `set_registered_model_alias(..., "dev")` — Stages sind in MLflow 3.x deprecated
  - [x] Zweiten Alias/Trennung `dev` → `prod` einführen — `scripts/promote_model.py` vergleicht Test-F1 von `dev` gegen `prod` und verschiebt den Alias nur bei Gleich- oder Verbesserung
  - [ ] Model Description je registriertem Modell setzen

- [~] **5.3** Deployment Pattern
  - [x] FastAPI lädt Model aus MLflow Registry (`models:/<name>@<alias>`)
  - [x] Artefakt-Download aus MinIO im Compose-Netz konfiguriert (S3-Endpoint, Credentials)
  - [x] Environment Variable für Model Alias/Version — `MODEL_ALIAS` (Default `dev`, Compose setzt `prod`)
  - [ ] Model Loading Funktion mit Fallback (letzte funktionierende Version)
  - [ ] Blue-Green Deployment Vorbereitung

- [ ] **5.4** Retraining Path  *(offen — TODO „Automatic retraining?")*
  - [ ] Manueller Retraining Trigger (API Endpoint `/model/retrain`)
  - [ ] Prefect Flow für Retraining (kapselt die `task_*_train_and_register.py`)
  - [ ] Retraining mit neuen Daten aus Trino
  - [ ] Auto-Register zu MLflow nach Retraining (Logik existiert bereits in den Scripts → wiederverwenden)
  - [x] Model Comparison (Old vs New) vor Alias-Umzug — `scripts/promote_model.py`, Vergleich per Test-F1
  - [x] Manuelle Promotion nach `prod` — `python scripts/promote_model.py --model <name>`, danach `docker compose restart model-service`
  - [ ] Trigger-Kriterium festlegen (Zeitplan vs. Evidently-Drift-Signal)

- [ ] **5.5** CI/CD für Modelle
  - [ ] GitHub Action für Model Training
  - [ ] Model Testing (Performance Thresholds, z.B. F1 ≥ Baseline)
  - [ ] Auto-Deploy zu Staging bei Success

---

### Phase 6: Monitoring

- [~] **6.1** Prometheus Service Monitoring
  - [x] `prometheus-fastapi-instrumentator` aktiviert, `/metrics` exponiert
  - [x] [prometheus/prometheus.yaml](../prometheus/prometheus.yaml) scraped `prometheus`, `model-service`, `evidently_service`
  - [ ] Custom Metrics hinzufügen:
    - [ ] Prediction Counter (by model type: pk/cpk/fk/cfk/normalform)
    - [ ] Prediction Duration Histogram (Model-Inferenz getrennt vom HTTP-Overhead)
    - [ ] Error Counter
    - [ ] Model Version Gauge
  - [ ] Prefect-Flows als Scrape-Target oder Pushgateway anbinden

- [~] **6.2** Golden Signals implementieren
  - [x] **Latency**: `http_request_duration_highr_seconds` (p95-Alert aktiv)
  - [x] **Traffic**: `http_requests_total` (No-Traffic-Alert aktiv)
  - [x] **Errors**: 4xx-/5xx-Rate-Alerts aktiv
  - [~] **Saturation**: nur `process_resident_memory_bytes` — CPU und Disk fehlen (z.B. node-exporter/cAdvisor ergänzen)

- [ ] **6.3** Grafana Dashboards  *(offen — TODO „Graphana Dashboard")*
  - [x] Grafana-Container in Compose inkl. Volume-Mounts für `provisioning/` und `dashboards/`
  - [ ] **`grafana/provisioning/` und `grafana/dashboards/` sind leer** → Datasource- und Dashboard-Provisioning anlegen
  - [ ] Dashboard für Golden Signals
  - [ ] Dashboard für Model Performance / Prediction-Verteilung
  - [ ] Dashboard für Data Pipeline (Queue-Länge, verarbeitete Zeilen)
  - [ ] Dashboard für MLflow Experiments

- [~] **6.4** Alerting
  - [x] [prometheus/alert.yaml](../prometheus/alert.yaml) mit 7 Regeln in 2 Gruppen:
    - [x] High Error Rate (5xx >5% / 4xx >25%)
    - [x] High Latency (p95 >500ms)
    - [x] Low Traffic + `InstanceDown` + `ModelServiceUnreachable`
    - [x] High Memory Usage (>1.5GB)
  - [x] Alertmanager konfiguriert und an Prometheus angebunden ([alertmanager.yml](../alertmanager/alertmanager.yml), UI auf :9093)
  - [ ] Notification Channels — Receiver `default` ist bewusst leer (null receiver); Slack/E-Mail via `*_file`-Secrets nachziehen
  - [ ] Einen Alert end-to-end testen und das Ergebnis dokumentieren

- [ ] **6.5** Evidently Model Monitoring
  - [x] Evidently-Service läuft als Container ([evidently_service/](../evidently_service/), Port 8085) und exponiert `/metrics`
  - [ ] **Service noch auf Tutorial-Daten**: [config.yaml](../evidently_service/config.yaml) und `green_taxi_data/reference.csv` gehören zum Taxi-Beispiel
  - [ ] Reference Dataset aus den echten PK/FK/Normalform-Features erzeugen
  - [ ] `column_mapping` auf die tatsächlichen Feature-Namen umstellen
  - [ ] Monitoring-Call in [app.py](../webservice/app.py) reaktivieren (Block ist auskommentiert)
  - [ ] Input Drift Detection + Prediction Quality Monitoring
  - [ ] Evidently-Metriken in Grafana visualisieren
  - [ ] Scheduled Reports mit Prefect

- [ ] **6.6** Logging & Tracing
  - [ ] Structured Logging (JSON) — Webservice nutzt aktuell `print()`/`traceback.print_exc()`
  - [ ] Log Aggregation (optional: Loki)
  - [ ] Correlation IDs für Request Tracing

---

### Phase 7: Betrieb, Doku & offene Entscheidungen *(neu)*

- [~] **7.1** Developer Experience
  - [x] Start-up-Script: [scripts/setup_stack.sh](../scripts/setup_stack.sh) — Preflight, Compose mit Health-Waits, Training (idempotent), Registry- und Schema-Verifikation, Smoke-Test aller 5 Endpoints, optional Streaming via `--with-streaming`
  - [x] `.env.template` / `.env.example` vorhanden
  - [ ] README um „Quickstart in 5 Minuten" ergänzen

- [ ] **7.2** Dokumentation aktuell halten
  - [ ] [flowchart.md](flowchart.md) an den Ist-Stand anpassen (Postgres/MinIO-Backend, Prefect statt dbt-Python-Models, zwei Pipelines)
  - [ ] Neue Docs in README verlinken: [FEATURE_PIPELINE_README.md](FEATURE_PIPELINE_README.md), [TRAIN_AND_PREDICT_FLOW.md](TRAIN_AND_PREDICT_FLOW.md), [NORMALFORM_PREDICTION.md](NORMALFORM_PREDICTION.md), [PYDANTIC_PREFECT_DBT_FIX.md](PYDANTIC_PREFECT_DBT_FIX.md), [INCREMENTAL_FEATURE_PROCESSING.md](INCREMENTAL_FEATURE_PROCESSING.md), [STREAMING_PREDICTION_APPROACHES.md](STREAMING_PREDICTION_APPROACHES.md)
  - [ ] Untracked Doku und `dbt/seeds/` committen

- [ ] **7.3** Offene Entscheidungen
  - [ ] Anderer Katalog als DuckDB für Queue und `prediction_results`? (Persistenz/Concurrency bei parallelen Flows)
  - [ ] Wo werden Features final extrahiert — Prefect (Ist-Zustand) oder zurück nach dbt/SQL?
  - [ ] Klären: `min_value` / `max_value` werden nur zum Gruppieren und für Traceability genutzt, nie als Feature — korrekt so? Falls ja, in [FEATURE_PIPELINE_README.md](FEATURE_PIPELINE_README.md) festhalten
  - [~] Inkrementelle Feature-Verarbeitung — Design steht, Umsetzung zurückgestellt (siehe [INCREMENTAL_FEATURE_PROCESSING.md](INCREMENTAL_FEATURE_PROCESSING.md))

---

### Nächste 5 sinnvolle Schritte

1. **Streaming end-to-end gegen Trino testen** (4.5) — Code steht, aber noch nie gegen die echte Datenbank gelaufen; erste echte Tabelle laden und den Lauf beobachten.
2. **Grafana-Dashboards provisionieren** (6.3) — die Metriken und Alerts liegen bereits an, es fehlt nur die Visualisierung.
3. **CI reparieren und härten** (2.1/2.3) — `dev`-Branch triggern, `[dev]`-Extra anlegen, `pytest --cov` aktivieren.
4. **Docker-Build-Workflow + Dockerfile-Härtung** (2.4/3.3) — Multi-stage, non-root, `.dockerignore`, Push zu GHCR.
5. **Retraining-Flow in Prefect** (5.4) — die Trainings-Scripts sind bereits registrierungsfähig, sie müssen nur orchestriert werden.

---

## 3. ARBEITSPAKETE MIT TEILSCHRITTEN

### ARBEITSPAKET 1: Foundation (Woche 1)

**Ziel**: Clean Code, Testing-Infrastruktur, Basis CI/CD

#### Teilschritte:
1. **Ruff Setup** (2 Stunden)
   - Installation: `uv add --dev ruff`
   - Konfiguration in `pyproject.toml`:
     ```toml
     [tool.ruff]
     line-length = 100
     target-version = "py311"
     select = ["E", "F", "I", "N", "W", "B", "Q"]
     ```
   - Pre-commit Hook: `.pre-commit-config.yaml`
   - Ersten Lint-Run: `ruff check . --fix`

2. **pytest Setup** (3 Stunden)
   - Installation: `uv add --dev pytest pytest-cov pytest-asyncio pytest-mock`
   - Verzeichnisstruktur:
     ```
     tests/
       conftest.py
       test_api/
       test_models/
       test_pipelines/
     ```
   - Erste Tests für FastAPI Endpoints schreiben

3. **GitHub Actions Basis** (2 Stunden)
   - `.github/workflows/ci.yml` erstellen
   - Jobs: lint, test
   - Badge zu README hinzufügen

**Deliverables**:
- ✅ Ruff läuft ohne Fehler
- ✅ Mindestens 10 Unit Tests mit >50% Coverage
- ✅ CI Pipeline läuft auf jedem Push

---

### ARBEITSPAKET 2: Docker & Registry (Woche 1-2)

**Ziel**: Production-ready Docker Images, GHCR Integration

#### Teilschritte:
1. **Dockerfile Optimierung** (3 Stunden)
   - Multi-stage Build
   - Beispiel:
     ```dockerfile
     FROM python:3.11-slim as builder
     WORKDIR /app
     COPY requirements.txt .
     RUN pip install --no-cache-dir -r requirements.txt

     FROM python:3.11-slim
     WORKDIR /app
     COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
     COPY . .
     RUN useradd -m appuser && chown -R appuser /app
     USER appuser
     HEALTHCHECK CMD curl --fail http://localhost:8080/ || exit 1
     CMD ["uvicorn", "webservice.app:app", "--host", "0.0.0.0", "--port", "8080"]
     ```

2. **GHCR Workflow** (2 Stunden)
   - `.github/workflows/docker-build.yml`
   - GitHub Secrets einrichten
   - Multi-arch Build (optional)

3. **Docker Compose Update** (1 Stunde)
   - Services auf GHCR Images umstellen
   - Image Versioning mit Tags

**Deliverables**:
- ✅ Docker Image <500MB
- ✅ Automatischer Push zu GHCR bei main-Branch
- ✅ Docker Compose nutzt GHCR Images

---

### ARBEITSPAKET 3: API Enhancement (Woche 2)

**Ziel**: Production-grade FastAPI Service

#### Teilschritte:
1. **Pydantic Models erweitern** (2 Stunden)
   - Models für PK, FK, Normalization
   - Validation Rules
   - Examples in Schema

2. **Endpoint-Erweiterungen** (3 Stunden)
   - `/api/v1/predict/pk`
   - `/api/v1/predict/fk`
   - `/api/v1/predict/normalization`
   - `/api/v1/health/live`
   - `/api/v1/health/ready`
   - `/api/v1/model/info`

3. **Error Handling** (1 Stunde)
   - Custom Exception Handlers
   - Structured Error Responses

**Deliverables**:
- ✅ OpenAPI Docs vollständig
- ✅ Alle Endpoints mit Tests
- ✅ Health Checks funktionieren

---

### ARBEITSPAKET 4: Data Pipeline Foundation (Woche 2-3)

**Ziel**: dbt + Prefect Basis-Setup

#### Teilschritte:
1. **dbt Projekt Setup** (3 Stunden)
   - Installation: `uv add dbt-core dbt-duckdb`
   - Projekt initialisieren: `dbt init alligator_metaswamp`
   - Seeds für Test-Daten erstellen
   - Erste Models schreiben

2. **Prefect Setup** (2 Stunden)
   - Installation: `uv add prefect`
   - Prefect Server starten: `prefect server start`
   - Ersten Flow erstellen

3. **Test Data Generation** (2 Stunden)
   - dbt Seeds mit synthetischen PK/FK Daten
   - Python Script für erweiterte Generation
   - Integration mit dbt

**Deliverables**:
- ✅ dbt läuft und generiert Test-Daten
- ✅ Prefect Server läuft lokal
- ✅ Erster Flow orchestriert dbt

---

### ARBEITSPAKET 5: Batch Pipeline (Woche 3)

**Ziel**: Batch Training und Prediction mit Prefect

#### Teilschritte:
1. **Batch Training Flow** (4 Stunden)
   - Prefect Flow Definition
   - Integration mit bestehendem Training-Code
   - MLflow Logging
   - Error Handling

2. **Batch Prediction Flow** (3 Stunden)
   - CSV Input Validation
   - Batch Prediction auf CSV
   - Output zu CSV/Database

3. **Scheduling** (1 Stunde)
   - Cron-basiertes Scheduling
   - Prefect Deployments

**Deliverables**:
- ✅ Training Flow läuft automatisch
- ✅ Batch Prediction akzeptiert CSV
- ✅ Flows sind schedulbar

---

### ARBEITSPAKET 6: Streaming Pipeline (Woche 3-4)

**Ziel**: Real-time Predictions via Events

#### Teilschritte:
1. **Message Queue Setup** (2 Stunden)
   - RabbitMQ/Redis Container zu docker-compose
   - Credentials konfigurieren

2. **Event Consumer** (4 Stunden)
   - Prefect Flow für Event Consumption
   - Message Parsing und Validation
   - Real-time Prediction
   - Result Publishing

3. **FastAPI Webhook** (2 Stunden)
   - Endpoint für Event Ingestion
   - Async Processing
   - OpenAPI Documentation

**Deliverables**:
- ✅ Events werden konsumiert
- ✅ Real-time Predictions funktionieren
- ✅ Webhook Endpoint getestet

---

### ARBEITSPAKET 7: MLflow Integration (Woche 4)

**Ziel**: Vollständiges Experiment Tracking und Model Registry

#### Teilschritte:
1. **Tracking in allen Scripts** (3 Stunden)
   - `mlflow.start_run()` in Training
   - Log Parameters, Metrics, Artifacts
   - Dataset Logging

2. **Model Registry** (2 Stunden)
   - Model Registration nach Training
   - Stages und Versioning
   - Metadata und Tags

3. **Production Model Loading** (2 Stunden)
   - FastAPI lädt Model aus Registry
   - Environment Variables für Version
   - Fallback Strategie

**Deliverables**:
- ✅ Alle Experimente in MLflow
- ✅ Models im Registry
- ✅ FastAPI nutzt Registry Model

---

### ARBEITSPAKET 8: Retraining Path (Woche 4)

**Ziel**: Manueller Retraining Workflow

#### Teilschritte:
1. **Retraining API Endpoint** (2 Stunden)
   - `/api/v1/model/retrain` POST
   - Trigger Prefect Flow

2. **Retraining Flow** (3 Stunden)
   - Load new data
   - Train model
   - Evaluate performance
   - Register to MLflow
   - Compare with production model

3. **Manual Promotion** ✅ erledigt — `scripts/promote_model.py` (F1-Gate `dev` vs. `prod`)
   - ~~Script für Stage Promotion~~
   - MLflow UI als Alternative

**Deliverables**:
- ✅ Retraining via API auslösbar
- ✅ Neues Model automatisch registriert
- ✅ Promotion-Prozess dokumentiert

---

### ARBEITSPAKET 9: Service Monitoring (Woche 4-5)

**Ziel**: Golden Signals mit Prometheus & Grafana

#### Teilschritte:
1. **Prometheus Metrics** (2 Stunden)
   - `prometheus-fastapi-instrumentator` aktivieren
   - Custom Metrics hinzufügen
   - Prometheus Config erweitern

2. **Golden Signals** (2 Stunden)
   - Queries für Latency, Traffic, Errors, Saturation schreiben
   - Recording Rules definieren

3. **Grafana Dashboard** (3 Stunden)
   - Dashboard JSON erstellen
   - Provisioning Setup
   - Visualisierungen optimieren

**Deliverables**:
- ✅ Alle 4 Golden Signals messbar
- ✅ Grafana Dashboard live
- ✅ Prometheus scraped alle Targets

---

### ARBEITSPAKET 10: Alerting (Woche 5)

**Ziel**: Mindestens 1 funktionaler Alert

#### Teilschritte:
1. **Prometheus Alert Rules** (2 Stunden)
   - `prometheus/alerts.yml` erstellen
   - High Error Rate Alert
   - High Latency Alert

2. **Alertmanager** (2 Stunden)
   - Alertmanager Container zu compose
   - Konfiguration für Notifications
   - Test Alerts senden

3. **Notification Channels** (1 Stunde)
   - Slack Webhook oder Email
   - Test Alert durchführen

**Deliverables**:
- ✅ Alerts feuern bei Conditions
- ✅ Notifications kommen an
- ✅ Alerting dokumentiert

---

### ARBEITSPAKET 11: Model Monitoring (Woche 5)

**Ziel**: Evidently für Drift und Quality

#### Teilschritte:
1. **Evidently Integration** (3 Stunden)
   - Reference Dataset definieren
   - Current Data Collection
   - Drift Detection konfigurieren

2. **Prediction Quality** (2 Stunden)
   - Metrics über Zeit tracken
   - Evidently Reports generieren

3. **Scheduled Reports** (2 Stunden)
   - Prefect Flow für Daily Reports
   - Reports zu Grafana (oder S3/File)

**Deliverables**:
- ✅ Drift Detection funktioniert
- ✅ Prediction Quality getrackt
- ✅ Reports verfügbar

---

## 4. GENERELLER ABLAUF

### Woche 1: Foundation

- **Tag 1-2**: Ruff Setup, Code Cleanup, Type Hints
- **Tag 3-4**: pytest Setup, erste Tests schreiben
- **Tag 5**: GitHub Actions CI, Docker Basis

### Woche 2: API & Containers

- **Tag 1-2**: FastAPI Endpoints erweitern, Pydantic Models
- **Tag 3-4**: Dockerfile optimieren, GHCR Pipeline
- **Tag 5**: dbt Setup, Prefect Installation

### Woche 3: Pipelines

- **Tag 1-2**: Batch Training und Prediction Flows
- **Tag 3-4**: Streaming Pipeline mit Message Queue
- **Tag 5**: Pipeline Testing und Debugging

### Woche 4: MLOps Core

- **Tag 1-2**: MLflow vollständige Integration
- **Tag 3-4**: Retraining Path implementieren
- **Tag 5**: Model Registry und Deployment Pattern

### Woche 5: Monitoring

- **Tag 1-2**: Prometheus Golden Signals, Custom Metrics
- **Tag 3-4**: Grafana Dashboards, Alerting
- **Tag 5**: Evidently Model Monitoring

---

## 5. TECHNOLOGIE-STACK ÜBERSICHT

| Kategorie | Technologie | Zweck |
|-----------|-------------|-------|
| **Code Quality** | Ruff | Linting & Formatting |
| **Testing** | pytest, pytest-cov | Unit & Integration Tests |
| **CI/CD** | GitHub Actions | Automation Pipeline |
| **Container** | Docker, Docker Compose | Containerization |
| **Registry** | GitHub Container Registry | Image Storage |
| **API** | FastAPI, Pydantic | Model Service |
| **Orchestration** | Prefect | Workflow Management |
| **Data Transform** | dbt | Data Pipeline |
| **Messaging** | RabbitMQ/Redis | Event Streaming |
| **ML Tracking** | MLflow | Experiment Tracking & Registry |
| **Service Monitoring** | Prometheus | Metrics Collection |
| **Visualization** | Grafana | Dashboards & Alerts |
| **Model Monitoring** | Evidently | Drift & Quality |
| **Logging** | Python logging, structlog | Application Logs |

---

## 6. ERFOLGSKRITERIEN

### Code Quality

- ✅ Ruff läuft ohne Errors
- ✅ 100% der Funktionen haben Type Hints
- ✅ >70% Test Coverage

### CI/CD

- ✅ Alle Commits werden getestet
- ✅ Docker Images werden automatisch gebaut
- ✅ Tests müssen grün sein vor Merge

### Data Pipeline

- ✅ Batch Training läuft automatisch
- ✅ Streaming Pipeline verarbeitet Events
- ✅ dbt generiert Test-Daten

### MLOps

- ✅ Alle Experimente in MLflow geloggt
- ✅ Produktions-Models im Registry
- ✅ Retraining manuell auslösbar

### Monitoring

- ✅ Alle 4 Golden Signals messbar
- ✅ Mindestens 1 Alert konfiguriert
- ✅ Evidently trackt Drift

---

## 7. NÄCHSTE SCHRITTE

1. **Sofort starten**: Arbeitspaket 1 (Foundation)
2. **Priorität hoch**: Arbeitspakete 2-3 (Docker, API)
3. **Parallel möglich**: Arbeitspakete 4-6 (Pipelines) und 7-8 (MLflow)
4. **Zum Schluss**: Arbeitspakete 9-11 (Monitoring)

---

## 8. HILFREICHE BEFEHLE

### Ruff

```bash
# Check
ruff check .

# Fix
ruff check . --fix

# Format
ruff format .
```

### pytest

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_api/test_predict.py -v
```

### Docker

```bash
# Build
docker build -t alligator-metaswamp:latest .

# Run
docker-compose up --build

# Push to GHCR
docker tag alligator-metaswamp:latest ghcr.io/username/alligator-metaswamp:latest
docker push ghcr.io/username/alligator-metaswamp:latest
```

### Prefect

```bash
# Start server
prefect server start

# Deploy flow
prefect deployment build flows/training.py:train_model -n training-daily

# Run flow
prefect deployment run train-model/training-daily
```

### dbt

```bash
# Run models
dbt run

# Test
dbt test

# Generate docs
dbt docs generate
dbt docs serve
```

### MLflow

```bash
# Start UI
mlflow ui --host 0.0.0.0 --port 5000

# List registered models
mlflow models list

# Promote model
mlflow models transition-stage --name pk-model --version 3 --stage Production
```

---

## 9. RESSOURCEN & DOKUMENTATION

- **Ruff**: https://docs.astral.sh/ruff/
- **pytest**: https://docs.pytest.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Prefect**: https://docs.prefect.io/
- **dbt**: https://docs.getdbt.com/
- **MLflow**: https://mlflow.org/docs/latest/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **Evidently**: https://docs.evidentlyai.com/

---

**Viel Erfolg bei der MLOps-Integration! 🚀**
