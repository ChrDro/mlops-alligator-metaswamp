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

### Phase 1: Clean, Structured Code (Woche 1)
- [x] **1.1** Ruff Installation und Konfiguration
  - [x] Ruff zu `pyproject.toml` hinzufügen
  - [x] Ruff-Konfiguration erstellen (line-length, rules, excludes)
  - [x] Pre-commit Hooks einrichten
  
- [x] **1.2** Code-Quality-Verbesserungen
  - [x] Bestehenden Code mit Ruff formatieren
  - [x] Type Hints zu allen Funktionen hinzufügen
  
- [ ] **1.3** Projektstruktur-Refactoring
  - [x] Klare Module-Trennung (data/, models/, api/, pipelines/)

---

### Phase 2: Testing & CI/CD (Woche 1-2)
- [ ] **2.1** pytest Setup
  - [ ] `pytest`, `pytest-cov`, `pytest-mock` installieren
  - [x] `tests/` Verzeichnis-Struktur erstellen
  - [ ] `conftest.py` für Fixtures einrichten
  - [ ] pytest.ini/pyproject.toml Konfiguration
  
- [ ] **2.2** Unit Tests schreiben
  - [ ] Tests für Data Processing Funktionen
  - [ ] Tests für Model Prediction Logik
  - [ ] Tests für FastAPI Endpoints (mit TestClient)
  - [ ] Tests für Daten-Validierung
  - [ ] Mindestens 70% Code Coverage
  
- [x] **2.3** GitHub Actions CI Pipeline
  - [x] `.github/workflows/ci.yml` erstellen
  - [x] Lint Job (Ruff Check)
  - [x] Test Job (pytest mit Coverage Report)
  - [ ] Matrix Testing (Python 3.11+)
  - [ ] Coverage Badge zu README hinzufügen
  
- [ ] **2.4** Docker Build & GHCR Pipeline
  - [ ] `.github/workflows/docker-build.yml` erstellen
  - [ ] Multi-stage Dockerfile optimieren
  - [ ] Docker Build und Push zu GHCR
  - [ ] Image Tagging (latest, SHA, semantic versioning)
  - [ ] GitHub Secrets für GHCR_TOKEN einrichten
  - [ ] Docker-compose.yml auf GHCR Images umstellen

---

### Phase 3: Model Service API & Docker (Woche 2)
- [ ] **3.1** FastAPI Service Verbesserungen
  - [ ] Pydantic Models für alle Endpoints (PK, FK, Normalization)
  - [ ] Request/Response Validation erweitern
  - [ ] Error Handling mit HTTPException
  - [ ] API Versioning (/api/v1/)
  - [ ] Health-Check Endpoint erweitern (readiness/liveness)
  - [ ] OpenAPI Schema dokumentieren
  
- [ ] **3.2** Model Loading Optimization
  - [ ] MLflow Model Loading beim Startup
  - [ ] Model Caching Strategie
  - [ ] Graceful Model Reload ohne Downtime
  - [ ] Model Version als Env Variable
  
- [ ] **3.3** Dockerfile Optimierung
  - [ ] Multi-stage Build (builder + runtime)
  - [ ] Layer Caching optimieren
  - [ ] Security: Non-root User
  - [ ] Minimal Base Image (python:3.11-slim)
  - [ ] .dockerignore erstellen
  - [ ] Health Check in Dockerfile

---

### Phase 4: Data Pipeline (Woche 2-3)
- [ ] **4.1** dbt Setup
  - [ ] dbt-core und dbt-duckdb installieren
  - [ ] dbt Projekt initialisieren (`dbt_project.yml`)
  - [ ] Profiles.yml für DuckDB konfigurieren
  - [ ] Seeds für Test-Daten erstellen
  - [ ] Models für Data Transformation
  
- [ ] **4.2** Test Data Generation
  - [ ] dbt Seed-Dateien für synthetische Daten
  - [ ] Python-Script zur erweiterten Datengenerierung
  - [ ] Datenqualitäts-Tests mit dbt tests
  - [ ] Schema-Dokumentation
  
- [ ] **4.3** Prefect Orchestration Setup
  - [ ] Prefect installieren (`prefect>=2.0`)
  - [ ] Prefect Server lokal starten
  - [ ] Flow für Batch Training erstellen
  - [ ] Flow für Batch Prediction erstellen
  - [ ] Flow für dbt Run orchestrieren
  
- [ ] **4.4** Batch Pipeline
  - [ ] CSV Batch Prediction Flow
  - [ ] CSV Input Validation
  - [ ] Batch Training Flow mit MLflow Logging
  - [ ] Error Handling und Retry Logic
  - [ ] Ergebnis-Export (CSV/Database)
  
- [ ] **4.5** Streaming Pipeline
  - [ ] Message Queue Setup (RabbitMQ/Kafka/Redis)
  - [ ] Prefect Flow für Event Processing
  - [ ] Event Consumer Implementation
  - [ ] FastAPI Webhook für Events
  - [ ] Real-time Prediction auf Events
  - [ ] Event Logging zu Monitoring

---

### Phase 5: MLOps Tracking, Versioning, Retraining (Woche 3-4)
- [ ] **5.1** MLflow Tracking vollständig integrieren
  - [ ] Experiment Tracking in allen Training-Scripts
  - [ ] Hyperparameter Logging (Optuna Integration)
  - [ ] Metrics Logging (Precision, Recall, F1)
  - [ ] Artifacts Logging (Confusion Matrix, Feature Importance)
  - [ ] Dataset Tracking mit MLflow Datasets
  
- [ ] **5.2** MLflow Model Registry
  - [ ] Model Registration nach Training
  - [ ] Model Stages (None → Staging → Production → Archived)
  - [ ] Model Versioning
  - [ ] Model Metadata (Description, Tags)
  - [ ] Model Signature und Input Example
  
- [ ] **5.3** Deployment Pattern
  - [ ] FastAPI lädt Model aus MLflow Registry
  - [ ] Environment Variable für Model Stage/Version
  - [ ] Model Loading Funktion mit Fallback
  - [ ] Blue-Green Deployment Vorbereitung
  
- [ ] **5.4** Retraining Path
  - [ ] Manueller Retraining Trigger (API Endpoint)
  - [ ] Prefect Flow für Retraining
  - [ ] Retraining mit neuen Daten
  - [ ] Auto-Register zu MLflow nach Retraining
  - [ ] Model Comparison (Old vs New)
  - [ ] Manual Promotion zu Production
  
- [ ] **5.5** CI/CD für Modelle
  - [ ] GitHub Action für Model Training
  - [ ] Model Testing (Performance Thresholds)
  - [ ] Auto-Deploy zu Staging bei Success

---

### Phase 6: Monitoring (Woche 4-5)
- [ ] **6.1** Prometheus Service Monitoring
  - [ ] prometheus-fastapi-instrumentator aktivieren
  - [ ] Custom Metrics hinzufügen:
    - [ ] Prediction Counter (by model type)
    - [ ] Prediction Duration Histogram
    - [ ] Error Counter
    - [ ] Model Version Gauge
  - [ ] Prometheus Konfiguration erweitern
  
- [ ] **6.2** Golden Signals implementieren
  - [ ] **Latency**: Request Duration (p50, p95, p99)
  - [ ] **Traffic**: Requests per Second
  - [ ] **Errors**: Error Rate (4xx, 5xx)
  - [ ] **Saturation**: CPU, Memory, Disk Usage
  
- [ ] **6.3** Grafana Dashboards
  - [ ] Dashboard für Golden Signals
  - [ ] Dashboard für Model Performance
  - [ ] Dashboard für Data Pipeline
  - [ ] Dashboard für MLflow Experiments
  - [ ] Grafana Provisioning (automatisches Dashboard-Setup)
  
- [ ] **6.4** Alerting
  - [ ] Prometheus Alerting Rules definieren:
    - [ ] High Error Rate (>5% in 5min)
    - [ ] High Latency (p95 >500ms)
    - [ ] Low Traffic (possible service down)
    - [ ] High Memory Usage (>80%)
  - [ ] Alertmanager konfigurieren
  - [ ] Notification Channels (Slack/Email)
  
- [ ] **6.5** Evidently Model Monitoring
  - [ ] Evidently Service erweitern
  - [ ] Input Drift Detection
  - [ ] Prediction Quality Monitoring
  - [ ] Data Quality Metrics
  - [ ] Evidently Reports zu Grafana
  - [ ] Scheduled Reports mit Prefect
  
- [ ] **6.6** Logging & Tracing
  - [ ] Structured Logging (JSON Format)
  - [ ] Log Aggregation (optional: Loki)
  - [ ] Correlation IDs für Request Tracing

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

3. **Manual Promotion** (1 Stunde)
   - Script für Stage Promotion
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
