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
│  - CSV Ingestion     │                      │  - Prefect Events    │
│  - Batch Training    │                      │  - Webhook + Cron    │
│  - Batch Prediction  │                      │  - Watermark Diff    │
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

> **Stand: 30. Juli 2026** — abgeglichen mit dem tatsächlichen Repo-Zustand
> (Branch `dev` auf `e170ae1`, Arbeitsverzeichnis sauber). Zahlen aus einem echten Lauf:
> `pytest --cov` → 273 passed / 86,38 %, `docker images`, `curl localhost:8080/metrics`
> gegen den laufenden Stack.
> Legende: `[x]` erledigt · `[~]` teilweise erledigt (Details im Unterpunkt) · `[ ]` offen · ~~durchgestrichen~~ = bewusst verworfen.

**Fortschritt auf einen Blick**

| Phase | Status | Kern-Lücke |
|-------|--------|------------|
| 1 Clean Code | 🟢 **100%** | keine offenen Punkte |
| 2 Testing & CI/CD | 🟡 ~75% | Tests stehen (273, 86 % Coverage); offen: CI horcht auf `develop` statt `dev`, `--cov` fehlt im Workflow, kein Docker-Build-Workflow, `main` steht auf dem Initial Commit |
| 3 API & Docker | 🟡 ~60% | Kein API-Versioning, keine echten Health-Probes, Dockerfile nicht gehärtet (root, single-stage) |
| 4 Data Pipeline | 🟡 ~70% | Streaming steht; dbt praktisch entkernt (keine Models/Seeds), Training nicht orchestriert |
| 5 MLflow & Retraining | 🟡 ~70% | `dev`→`prod` Promotion mit F1-Gate steht; Retraining selbst (Trigger, Prefect-Flow) offen |
| 6 Monitoring | 🟢 ~85% | Custom Model-Metriken fehlen, Alert-Receiver leer, kein Pipeline-Dashboard |
| 7 Betrieb & Doku | 🔴 ~30% | **README beschreibt Dateien, die es nicht gibt**; 7 Begleit-Dokumente nie committet |

**Seit dem letzten Abgleich (27. → 29. Juli) neu dazugekommen**

| Was | Wo | Effekt auf den Plan |
|-----|----|---------------------|
| Pre-commit inkl. gitleaks | [.pre-commit-config.yaml](../.pre-commit-config.yaml) | 1.1 geschlossen |
| Evidently produktiv auf **5 echten Tracks** | [evidently_service/](../evidently_service/), [config.yaml](../evidently_service/config.yaml) | 6.5 fast geschlossen |
| Referenz-/Holdout-Generator aus den Pydantic-Modellen | [build_monitoring_references.py](../evidently_service/build_monitoring_references.py) | 6.5, neuer Contract |
| Drift-Forwarding aus dem Serving-Pfad | [monitoring_client.py](../webservice/monitoring_client.py) + `BackgroundTasks` | 3.1/6.5 geschlossen |
| Model-Quality-Backtest als Prefect-Flow (stündlich) | [model_quality_backtest.py](../prefect/model_quality_backtest.py) | 6.5, Basis für 5.4-Trigger |
| 5 provisionierte Grafana-Dashboards + Datasource | [grafana/](../grafana/) | 6.3 fast geschlossen |
| Prefect-Deployments von 3 auf 4 (Backtest-Cron) | [serve_flows.py](../prefect/serve_flows.py) | 4.3 |
| `predict_proba` auf `max(proba)` umgestellt | [predict.py:103](../webservice/predict.py#L103) | Multiclass-Bug in 3.2 |
| `.env.template` entfernt, `.env.example` ist Single Source | [.env.example](../.env.example) | 7.1 |
| dbt-Seeds **entfernt** (`ad3ccdf`) | [dbt/](../dbt/) | 4.1/4.2 zurückgefallen |
| `dev`→`prod` Promotion mit F1-Gate | [promote_model.py](../scripts/promote_model.py) | 5.2/5.4 (Model Comparison + Manuelle Promotion) geschlossen |

---

### Phase 1: Clean, Structured Code

- [x] **1.1** Ruff Installation und Konfiguration
  - [x] Ruff zu `pyproject.toml` hinzufügen (gepinnt auf `0.16.0`)
  - [x] Ruff-Konfiguration erstellen → eigene [`ruff.toml`](../ruff.toml) (line-length, rules, excludes inkl. `README.md`)
  - [x] **Pre-commit Hooks eingerichtet** ([.pre-commit-config.yaml](../.pre-commit-config.yaml)): `ruff --fix` + `ruff-format` (rev auf `v0.16.0` gepinnt, synchron zu `pyproject.toml`/CI), dazu `check-added-large-files`, `check-yaml/toml/json`, `check-ast`, `detect-private-key`, `check-merge-conflict`, `no-commit-to-branch=main`, `trailing-whitespace`, `end-of-file-fixer`, `requirements-txt-fixer` — plus **gitleaks** (`v8.18.2`) als Secret-Scanner
  - [x] **`pytest`-Hook aktiviert (30.07.)** — der auskommentierte Block zeigte auf ein nie existierendes `tests/`-Verzeichnis. Jetzt aktiv als `pytest-fast` **ohne Pfadargument**: `testpaths` in `pyproject.toml` ist die einzige Stelle, an der der Pfad steht, also kann er nicht wieder auseinanderlaufen. Läuft mit `-m "not integration" --tb=short -x` in **1,2 s** (46 Tests, 10 stack-abhängige deselektiert). Bewusst kein Gate: `--no-verify` umgeht ihn und er existiert nur dort, wo `pre-commit install` gelaufen ist — das Gate sind CI auf `dev`/`main` plus Branch Protection (2.3). Der Hook liefert nur früheres Feedback. Ebenso bewusst auf der `pre-commit`-Stage und nicht `pre-push`: `pre-commit install` installiert **nur** den pre-commit-Hook, ein `stages: [pre-push]`-Hook würde also für jeden, der den Standardbefehl benutzt hat, unsichtbar nie feuern

- [x] **1.2** Code-Quality-Verbesserungen
  - [x] Bestehenden Code mit Ruff formatiert (`ruff check` + `ruff format` grün in CI)
  - [x] Type Hints in Training-Scripts, Prefect-Flows und Webservice

- [x] **1.3** Projektstruktur-Refactoring — **abgeschlossen 30.07.**
  - [x] Klare Module-Trennung: [task_1/](../task_1/) (PK/CPK), [task_2/](../task_2/) (FK/CFK), [task_3/](../task_3/) (Normalform), [prefect/](../prefect/), [webservice/](../webservice/), [evidently_service/](../evidently_service/), [grafana/](../grafana/), [test/](../test/)
  - [x] Pipeline-Dateien umbenannt (`pk_fk_pipeline.py`, `normalform_pipeline.py`)
  - [x] `mlflow.db` und `mlruns/` sind jetzt in [.gitignore](../.gitignore) (Zeile 221/222) — lokale Altlasten liegen noch im Arbeitsverzeichnis, gehen aber nicht mehr ins Repo
  - [x] **`src/test_trino_connection.py` → [`src/check_trino_connection.py`](../src/check_trino_connection.py) (30.07.)** — der `test_`-Prefix ließ pytest die Datei sammeln, und weil der Verbindungsaufbau auf Modulebene stand, **öffnete jeder `pytest`-Lauf eine echte Trino-Verbindung während der Collection**. Der `except DBAPIError` schluckte den Fehler, also sah nichts davon verdächtig aus. Alles liegt jetzt hinter `main()` — Import ist nebenwirkungsfrei. Bewusst in `src/` geblieben statt nach `test/` verschoben: es ist ein manuelles Diagnose-Skript, kein Test. Zweite Absicherung von der anderen Seite: `testpaths = ["test"]` (2.1) hält die Collection ohnehin aus `src/` heraus
  - [x] ~~Untracked Artefakte im Repo-Root aufräumen~~ — **erledigt/entfallen (30.07.)**: die vier `midterm_presentation*`-Dateien sind weg (`presentation/` ist ein leeres Restverzeichnis), `mlflow.log` und `logs/dbt.log` liegen noch lokal, greifen aber unter `*.log` in [.gitignore](../.gitignore) und können das Repo nicht erreichen. Einzige Ausnahme ist `workflow_idea.txt` — das ist kein Aufräum-, sondern ein Inhaltsthema und gehört zur dbt-Grundsatzentscheidung in 4.1(a), wo es ohnehin referenziert wird

---

### Phase 2: Testing & CI/CD

- [~] **2.1** pytest Setup
  - [x] `pytest` ist Hauptabhängigkeit; **`pytest-cov` (7.1.0), `pytest-mock` (3.15.1) und `httpx2` (für `TestClient`) am 30.07. nachgezogen** — bewusst ebenfalls in `dependencies`, nicht in einem Extra: ein blankes `uv sync` soll linten, testen **und** Coverage messen können, ohne dass sich jemand ein Flag merken muss. Preis ist, dass ein Consumer das Test-Tooling mitinstalliert — bewusst in Kauf genommen. `uv.lock` regeneriert
  - [x] Test-Verzeichnisstruktur: [test/test_data/](../test/test_data/) (Datenqualität) und [test/test_models/](../test/test_models/) (Schema-Contract)
  - [x] `conftest.py` je Paket: [test_data/conftest.py](../test/test_data/conftest.py), [test_models/conftest.py](../test/test_models/conftest.py)
  - [x] **273 Tests, alle grün** — von 56 am 29.07. auf 273 am 30.07.; Laufzeit 6,5 s mit Coverage, 3,4 s auf dem schnellen Pfad
  - [x] **pytest-Konfiguration in `pyproject.toml` (30.07.)**: `testpaths = ["test"]`, `addopts = "--strict-markers"`, Marker `integration`. Dazu `[tool.coverage.run]`/`[tool.coverage.report]` mit `fail_under = 80`, damit lokal und in CI dieselben Zahlen und dieselbe Schwelle gelten. `--cov` steht **nicht** in `addopts`: das würde jeden lokalen Lauf instrumentieren und die Fehlerausgabe unnötig zumüllen → CI übergibt die Coverage-Flags explizit
    - Reichweite von `--strict-markers`, damit es nicht überschätzt wird: es macht einen **nicht registrierten Marker-Dekorator** zum Collection-Error (`@pytest.mark.integraton` → `'integraton' not found in markers configuration option`, verifiziert). Es prüft **keine `-m`-Ausdrücke** — `-m "not integraton"` selektiert weiterhin alles. Letzteres fällt aber von selbst auf, weil dann die stack-abhängigen Tests mitlaufen
    - Stolperstein für später: `source` in `[tool.coverage.run]` akzeptiert nur **Verzeichnisse oder importierbare Pakete**, keine Dateipfade. Ein Eintrag `prefect/change_events.py` wird still ignoriert (`Module … was never imported`). Deshalb steht dort `prefect` als Verzeichnis, und die stack-abhängigen Pipeline-Module werden über `omit` ausgenommen. Coverage bevorzugt das lokale `./prefect`-Verzeichnis gegenüber der installierten `prefect`-Library — aber nur, wenn aus dem Repo-Root gestartet wird
  - [x] **Kein `dev`-Extra mehr, alles in `dependencies` (30.07.)** — das gesamte Test-Tooling (`pytest`, `pytest-cov`, `pytest-mock`, `httpx2`, `ruff`, `pre-commit`) steht in `dependencies`, damit ein blankes `uv sync` linten, testen **und** Coverage messen kann, ohne dass sich jemand ein Flag merken muss. Nachprüfbar: `uv export` und `uv export --extra dev` liefern identische Paketmengen. Preis ist, dass ein Consumer das Test-Tooling mitinstalliert — bewusst in Kauf genommen
    - `[project.optional-dependencies] dev = []` bleibt als **leerer Alias** stehen, damit CIs `uv pip install -e ".[dev]"` weiter auflöst. Ein unbekanntes Extra würde uv nur **warnen** statt zu scheitern — genau so blieb vorher unbemerkt, dass `pytest-cov` fehlte und die Codecov-/HTML-Upload-Steps nie etwas zu laden hatten. Alias und CI-Zeile gemeinsam in 2.3 entfernen

- [x] **2.2** Unit Tests schreiben — **erledigt am 30.07.**, 56 → 273 Tests, 86 % Coverage im gemessenen Scope. ⚠️ Der Verweis auf `TEST_SUITE.md` ist ein toter Link: `documentation/` enthält nur diese Datei und `flowchart.md`. Das Dokument entweder anlegen oder den Verweis entfernen — sonst wiederholt sich das Muster der 7 nie committeten Begleit-Dokumente (7.2)
  - [x] Datenqualitäts-Tests für Trainingsdaten: [test_training_data_quality.py](../test/test_data/test_training_data_quality.py), [..._nf.py](../test/test_data/test_training_data_quality_nf.py), [..._subject_area.py](../test/test_data/test_training_data_quality_subject_area.py)
  - [x] Schema-Contract-Tests über alle 5 Modelle: [test_model_schema_contract.py](../test/test_models/test_model_schema_contract.py) (Menge **und** Reihenfolge der Features, MLflow-Signatur vs. Pydantic)
  - [x] **Prediction-Logik**: [test/test_webservice/test_predict.py](../test/test_webservice/test_predict.py) — 12 Tests, MLflow-Model durch `FakePyFuncModel` ersetzt (behält die Trennung pyfunc ↔ rohes sklearn-Modell, weil genau die der 27.07.-Bug war). Der Multiclass-Test ist so gebaut, dass altes `probabilities[0][1]` (0,20) und korrektes `max(proba)` (0,55) **verschiedene** Zahlen liefern — ein wiedereingeführter fixer Klassenindex fällt sofort auf. `_align_to_signature` in beide Richtungen: Umsortierung *und* Fehlermeldung mit Feldnamen
  - [x] **FastAPI Endpoints**: [test_api_endpoints.py](../test/test_webservice/test_api_endpoints.py) — 75 Tests. Alle 5 Routen ×: 200 + Prediction/Probability, Echo aller Request-Features, **richtiger Modellname**, **eigener Monitoring-Track**, Feature-Reihenfolge an `predict()`, 400 bei Predict-Fehler, 422 bei unvollständigem/untypisiertem Body, Normalisierung aller Prediction-Shapes. Dazu `GET /`, `/metrics` und `POST /events/new-data` (202, Default-Schema, 503 mit Cron-Hinweis). Die fünf Handler sind Copy-Paste-Klone (3.1) — deshalb ist *jede* Zusicherung über alle fünf parametrisiert, nicht nur an einem Beispiel geprüft
  - [x] **Pydantic-Validierung**: [test_data_models.py](../test/test_webservice/test_data_models.py) — 39 Tests über alle 5 Request/Response-Paare: kein Feld hat einen Default (ein stiller Default würde als echte Messung gescort), fehlende/untypisierte Werte werden abgelehnt, unbekannte Felder werden **ignoriert** (dokumentiert, weil `predict_batch` darauf baut), Response = Request + `prediction` + `probability`. Plus die `schema`/`schema_name`-Alias-Mechanik der Event-Modelle
  - [x] **Prefect-Tasks**: [test/test_pipelines/](../test/test_pipelines/) — 70 Tests, kein Trino nötig. `diff_snapshots` in allen Fällen inkl. des **dokumentierten blinden Flecks** (In-place-UPDATE bei gleicher Zeilen-/Spaltenzahl ist unsichtbar); Claim-Protokoll gegen eine `FakeConnection`, die das erzeugte SQL mitschreibt: Track-Allowlist greift **vor** dem Execute, `complete`/`release` fassen nur den eigenen `run_id` an, Tabellennamen sind gebundene Parameter (mit einem Namen getestet, der interpoliert aus dem String-Literal ausbrechen würde); `recompute_table_stats` inkl. der Cross-Column-Features, die sich selbst ausschließen müssen
  - [x] **`monitoring_client.py`**: [test_monitoring_client.py](../test/test_webservice/test_monitoring_client.py) — 11 Tests, 100 % Coverage. Jeder Fehlerpfad einzeln: Connection-Error, Timeout, generische RequestException, 4xx/5xx über `raise_for_status`, **und** eine Nicht-`requests`-Exception. Dazu die Log-Level-Entscheidung selbst (Transportfehler auf `debug`, Unerwartetes auf `error` mit Traceback), weil die der Grund für den doppelten `except` ist. Bonus: [test_event_publisher.py](../test/test_webservice/test_event_publisher.py) (10 Tests) für die *entgegengesetzte* Zusage — dieser Pfad muss werfen
  - [x] **Coverage: 86 %** im gemessenen Scope (Ziel war 70 %) — `webservice/` **99 %** (`predict.py`, `monitoring_client.py`, `event_publisher.py`, alle 5 Datenmodelle je 100 %; `app.py` 96 %), die zwei getesteten Pipeline-Module 58 %. Gate steht auf `fail_under = 80` in `[tool.coverage.report]`, gilt damit lokal und in CI

- [~] **2.3** GitHub Actions CI Pipeline
  - [x] [.github/workflows/ci.yml](../.github/workflows/ci.yml) mit Lint-, Test- und Summary-Job
  - [x] Lint Job (`ruff check` + `ruff format --check`, Version im Workflow gepinnt und kommentiert)
  - [x] Trigger deckt `feature/**`, `hotfix/**`, `bugfix/**` ab → die Feature-Branches, auf denen tatsächlich gearbeitet wird, lösen CI aus
  - [~] Test Job läuft `pytest` — **ohne `--cov`**, die Codecov-/HTML-Upload-Steps laden daher nie existierende Artefakte hoch (`coverage.xml`, `htmlcov/`)
  - [~] Matrix Testing vorhanden, aber nur `['3.11']` → um 3.12 erweitern oder Matrix entfernen
  - [ ] Coverage Badge zu README hinzufügen
  - [ ] `push`/`pull_request` horchen auf `develop`; der Integrationsbranch heißt `dev` → direkte Pushes auf `dev` und PRs *nach* `dev` laufen ohne CI. Entweder `develop` → `dev` umbenennen oder beides listen
  - [ ] `origin/main` steht auf `e203865 Initial commit`, `origin/dev` ist **100 Commits** weiter → der Branch-Schutz („Tests grün vor Merge") greift heute faktisch nirgends; `dev` → `main` mergen und `main` als Default-Ziel etablieren

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
  - [x] Smoke-Tests per curl: [curl_tests/](../curl_tests/) — 5 Predict-Skripte. Der Streaming-Trigger heißt seit dem 30.07. [trigger_prefect_pipeline.sh](../trigger_prefect_pipeline.sh) und liegt im **Repo-Root**, nicht in `curl_tests/` → entweder zu den anderen Skripten zurückverschieben oder in der README als eigener Einstiegspunkt dokumentieren
  - [x] `POST /events/new-data` als Push-Trigger für die Streaming-Pipeline (202 / 503 mit Hinweis auf den Cron-Fallback)
  - [x] **Drift-Forwarding aus dem Serving-Pfad aktiv**: jeder der 5 Endpoints hängt `forward_to_monitoring(response, "<track>")` an `BackgroundTasks`, d.h. nach dem Senden der Response. Der frühere auskommentierte Evidently-Block ist damit ersetzt, nicht nur reaktiviert
  - [ ] API Versioning (`/api/v1/…`) — Code nutzt weiterhin flache Pfade
  - [ ] Health-Check erweitern: `/health/live` + `/health/ready` (heute nur `GET /` mit statischer Message; ein `ready`, das die MLflow-Erreichbarkeit prüft, würde die Startup-Latenz aus 3.2 sichtbar machen)
  - [ ] `/model/info` Endpoint (Modellname, Version, Alias)
  - [ ] OpenAPI-Schema anreichern (Beschreibungen, `examples`, Tags) — der Modul-Docstring in [app.py](../webservice/app.py) beschreibt noch „POST /predict" aus dem Tutorial statt der 5 echten Routen
  - [ ] `print()`-Debugausgaben in jedem Endpoint (`"Sending the following columns…"`) durch Logging ersetzen (siehe 6.6)
  - [ ] Die fünf Predict-Handler sind bis auf Modellname und Track identisch (~35 Zeilen ×5) → auf einen gemeinsamen Helper ziehen; jede Änderung muss heute fünfmal gemacht werden

- [~] **3.2** Model Loading Optimization
  - [x] Model Caching über `@lru_cache(maxsize=5)` in [predict.py](../webservice/predict.py)
  - [x] **Bugfix 27.07.:** `predict_proba` lief am rohen sklearn-Modell mit unsortierten Spalten, während `model.predict()` über pyfunc umsortiert. Sobald die Pydantic-Feldreihenfolge von der Trainingsreihenfolge abwich (cpk, cfk), scheiterte jeder Call mit `ValueError: feature names should match`. `_align_to_signature()` richtet die Spalten jetzt einmal an der Modellsignatur aus und meldet Schema-Drift mit Feldnamen statt einer nichtssagenden sklearn-Meldung
  - [x] **Bugfix 27.07.:** `ForeignKey` in [data_model_fk.py](../webservice/data_model_fk.py) war nicht mit `fk_model` synchron (4 Felder fehlten, 5 überzählig). Ursache: das fk-Trainingsskript wählt Features per **Drop-Liste**, das Pydantic-Schema wurde bei einer Erweiterung nicht nachgezogen
  - [x] **Schema-Drift-Contract-Test:** [test/test_models/test_model_schema_contract.py](../test/test_models/test_model_schema_contract.py) vergleicht für alle 5 Modelle die MLflow-Signatur gegen die Pydantic-Felder — Menge **und** Reihenfolge, getrennt geprüft. Hat sofort die verbliebene Reihenfolgen-Drift in `CompositePrimaryKey` und `CompositeForeignKey` gefunden (beide korrigiert). Skippt sauber, wenn kein MLflow erreichbar ist; [conftest.py](../test/test_models/conftest.py) fällt für nicht auflösbare Compose-Hostnames automatisch auf `localhost` zurück, sodass derselbe Test auf Host und im Container läuft
  - [x] **Bugfix 28.07.:** die Confidence war `probabilities[0][1]` — ein hart kodierter Klassenindex, der beim 4-klassigen Normalform-Modell schlicht die Wahrscheinlichkeit von Klasse 1 zurückgab. Jetzt `max(probabilities[0])`, also die Wahrscheinlichkeit der *vorhergesagten* Klasse; korrekt für binär und multiclass ([predict.py:103](../webservice/predict.py#L103))
  - [ ] Contract-Test in CI aktivieren — läuft dort mangels MLflow-Stack aktuell nur als Skip; sinnvoll wäre ein Job, der den Compose-Stack hochfährt
  - [ ] Model Loading beim Startup (heute lazy beim ersten Request → erster Call ist langsam)
  - [ ] Graceful Model Reload ohne Downtime (Cache-Invalidierung / Reload-Endpoint). Der Startpunkt ist `load_model.cache_clear()` auf dem `@lru_cache`; die README behauptet bereits einen `reload_models()`-Hook, den es nicht gibt
  - [ ] Model Alias/Version als Env Variable — `alias = "dev"` ist in [predict.py:14](../webservice/predict.py#L14) hart kodiert
  - [ ] `load_dotenv()` und `mlflow.set_tracking_uri()` aus dem Request-Pfad in den Startup ziehen (laufen heute bei **jedem** Predict)

- [~] **3.3** Dockerfile Optimierung ([webservice/Dockerfile](../webservice/Dockerfile))
  - [x] Minimal Base Image (`python:3.11.13-slim-bookworm`, `libgomp1` für LightGBM)
  - [ ] Multi-stage Build (builder + runtime)
  - [ ] Layer Caching optimieren — `COPY . /app` steht **vor** `pip install`, jede Code-Änderung invalidiert also die komplette Dependency-Installation
  - [ ] Security: Non-root User
  - [ ] `.dockerignore` erstellen — existiert **weder** im Root noch in `webservice/` oder `evidently_service/`; aktuell wandern `__pycache__/` und alles andere aus dem Build-Kontext ins Image
  - [ ] `HEALTHCHECK` im Dockerfile (`model-service` ist der einzige Service in Compose ohne Healthcheck, deshalb kann nichts sinnvoll `depends_on: service_healthy` darauf setzen)
  - [ ] Gleiches Härtungspaket für [evidently_service/Dockerfile](../evidently_service/Dockerfile) — dort zusätzlich: `COPY . /app` backt die Referenz-CSVs ins Image, weshalb [setup_stack.sh](../scripts/setup_stack.sh) den Service nach jedem Baseline-Bau neu bauen muss. Ein Volume-Mount für `references/` würde diesen Rebuild-Schritt ersatzlos streichen

---

### Phase 4: Data Pipeline

- [~] **4.1** dbt Setup — **⚠️ faktisch entkernt, Entscheidung überfällig**
  - [x] `dbt-core` + **`dbt-trino`** installiert (statt dbt-duckdb: DuckDB wird als Trino-Katalog angesprochen)
  - [x] dbt Projekt initialisiert: [dbt/dbt_project.yml](../dbt/dbt_project.yml)
  - [x] [profiles.yml](../dbt/profiles.yml) für Trino/LDAP konfiguriert (TODO „change dbt profile" erledigt)
  - [x] `target/` (Zeile 76) und `dbt/logs/` (Zeile 225) sind in [.gitignore](../.gitignore)
  - [x] ~~Models für Data Transformation~~ — bewusst verworfen: die Feature-Extraktion liegt in Prefect, weil dbt-trino keine Python-Models unterstützt (siehe Docstring in [pk_fk_pipeline.py:10](../prefect/pk_fk_pipeline.py#L10))
  - [x] ~~Seeds für Test-Daten~~ — mit `ad3ccdf` („Removed queue.csv and yaml as dbt seeds") **entfernt**
  - [ ] **Ist-Zustand: von dbt sind nur noch `dbt_project.yml`, `profiles.yml` und `.user.yml` getrackt.** Keine Models, keine Seeds, keine Tests, kein Aufruf aus Prefect, Compose oder `setup_stack.sh`. Der einzige verbleibende Bezug im Code ist ein erklärender Kommentar. Damit ist dbt derzeit reine Konfiguration ohne Wirkung — eine der beiden Richtungen muss gewählt werden:
    - **(a) reaktivieren** entlang der Demo-Idee: `dbt` lädt `tpch.customer` als CSV-Seed nach `new_predict_data` in Trino, das triggert per Watermark-Diff die Streaming-Pipeline. Das gäbe dem Demo-Pfad ein realistisches Ingestion-Frontend und würde „dbt" im Stack ehrlich machen
      - ⚠️ Diese Idee stand nur in einer untracked `workflow_idea.txt` im Repo-Root, die beim Aufräumen am 30.07. **gelöscht** wurde — sie ist damit nur noch hier festgehalten. Ebenso sind die zwischenzeitlich angelegten `dbt/seeds/*.csv` und `dbt/test/` wieder verschwunden. Wenn (a) gewählt wird, gehören Seeds und Demo-Beschreibung diesmal ins Repo, nicht ins Arbeitsverzeichnis
    - **(b) entfernen** und `dbt-core`/`dbt-trino`/`prefect-dbt` aus `pyproject.toml` streichen — drei nicht genutzte Abhängigkeiten samt Transitivem
  - [ ] Solange (a) nicht umgesetzt ist: die Zeile „Data pipeline | Prefect + dbt" in der README-Stack-Tabelle korrigieren (Prefect ist fertig, dbt nicht vorhanden)

- [~] **4.2** Test Data Generation
  - [x] Schema-/Datendokumentation: [test/test_data/README.md](../test/test_data/README.md), `README_NF.md`, `README_SubjectArea.md`
  - [x] Datenqualitätstests — als **pytest**-Suite umgesetzt (nicht als dbt tests), läuft gegen die CSVs in [data/](../data/)
  - [x] Holdout-/Referenz-Generierung für das Monitoring: [build_monitoring_references.py](../evidently_service/build_monitoring_references.py) splittet die gelabelten Trainingsdaten in Referenz + Holdout, stratifiziert je Zielspalte
  - [x] ~~dbt Seed-Dateien~~ — entfernt, siehe 4.1
  - [ ] Python-Script zur erweiterten Datengenerierung (synthetische Tabellen mit bekannten PK/FK/Normalform-Eigenschaften) — bisher hängt alles am eingefrorenen, handgelabelten Datensatz

- [~] **4.3** Prefect Orchestration Setup
  - [x] Prefect 3.7 installiert (`prefect`, `prefect-sqlalchemy`, `prefect-dbt[trino]`)
  - [x] Prefect Server als Service in [docker-compose.yaml](../docker-compose.yaml) (Port 4200)
  - [x] Flow für Batch Prediction: `key-prediction-pipeline` in [pk_fk_pipeline.py](../prefect/pk_fk_pipeline.py)
  - [x] Flow für Normalform-Prediction: `normalform-prediction-pipeline` in [normalform_pipeline.py](../prefect/normalform_pipeline.py) (TODO „predict normalform into prefect flow" erledigt)
  - [x] **Prefect Deployments + Scheduling erledigt** — [serve_flows.py](../prefect/serve_flows.py) registriert **4 Deployments** über `serve()`, jeweils mit `ConcurrencyLimitConfig` und bewusst gewählter Kollisionsstrategie:

    | Deployment | Auslöser | Strategie | Begründung im Code |
    |---|---|---|---|
    | `change-detection-check/streaming` | Event `alligator.new-data.arrived` + Cron `*/15 * * * *` | `CANCEL_NEW` | Ein laufender Snapshot sieht den aktuellen Stand schon; Bursts werden verworfen |
    | `key-prediction-pipeline/streaming` | Event `alligator.changes.recorded` | `ENQUEUE` | Ein verworfener Lauf könnte `pending_changes` stranden lassen |
    | `normalform-prediction-pipeline/streaming` | dasselbe Event (parallel) | `ENQUEUE` | dito; keine der beiden Pipelines kann die andere aushungern |
    | `model-quality-backtest/scheduled` | Cron `17 * * * *` | `CANCEL_NEW` | Überlappende Läufe würden ihre Zeilen im selben Evidently-Fenster vermischen |
  - [x] Trigger hängen an den Deployments selbst, werden also bei jedem Prozessstart mit-synchronisiert — es gibt keinen separaten „Automations anlegen"-Schritt, der vergessen werden kann
  - [ ] Flow für Batch **Training** — die fünf `task_*_train_and_register.py` laufen weiterhin standalone
  - [x] ~~Flow für dbt Run orchestrieren~~ — entfällt, solange keine dbt-Models existieren (siehe 4.1)

- [~] **4.4** Batch Pipeline
  - [x] Feature-Extraktion aus `information_schema` → `staging.stg_column_features`
  - [x] **Bugfix 30.07. — `null_ratio_rank` in [pk_fk_pipeline.py:261](../prefect/pk_fk_pipeline.py#L261)**: das Extraktions-SQL rankt nach `null_ratio ASC, ordinal_position ASC`, der Python-Recompute, der diesen Wert nach dem Batching **überschreibt**, sortierte nur nach `ordinal_position`. Damit bedeutete `null_ratio_rank` faktisch „Position in der Tabelle" und `is_least_null_in_table` „ist die erste Spalte" — unabhängig von Nulls. `null_ratio` lag die ganze Zeit im DataFrame, der Sortierschlüssel war schlicht weggefallen. Gefunden beim Schreiben der Tests in 2.2, nicht durch einen Fehler: es hat nie etwas geworfen, das Feature hat nur aufgehört zu bedeuten, was sein Name sagt
    - Nur `pk_fk_pipeline.py` war betroffen. [normalform_pipeline.py:381](../prefect/normalform_pipeline.py#L381) sortierte bereits korrekt und diente als Referenzimplementierung → **Retrain betrifft die vier Key-Modelle (pk/cpk/fk/cfk) und deren Evidently-Referenzen, nicht das Normalform-Modell**
    - Zwei Regressionstests in [test_feature_recompute.py](../test/test_pipelines/test_feature_recompute.py) (Ranking nach `null_ratio`, Tie-Break nach Position) — vorher als `xfail(strict=True)` dokumentiert, jetzt aktiv
  - [x] Prediction-Queue + `fetch_new_rows` / `predict_batch` / `store_predictions_to_trino`
  - [x] Aggregation & Speicherung der Normalform-Ergebnisse in eigener Tabelle
  - [x] Error Handling und Retry Logic (`retries=2`, `retry_delay_seconds`)
  - [x] Keine „rows processed"-Meldung mehr bei leeren Predictions (TODO erledigt)
  - [x] Ergebnis-Export nach Trino: `duckdb.prediction_results.key_results` / `nf_results`
  - [ ] Dedizierter CSV-Batch-Prediction-Flow inkl. Input-Validation
  - [ ] Training-Flow mit MLflow Logging (siehe 4.3)

- [~] **4.5** Streaming Pipeline — **umgesetzt am 27.07.2026**
  - [x] Optionen abgewogen und entschieden: Push-Webhook **+** Watermark-Polling als Sicherheitsnetz, Prefect Events als Transport (die Begründung steht heute nur noch im Modul-Docstring von [serve_flows.py](../prefect/serve_flows.py) — das Entscheidungsdokument wurde nie committet, siehe 7.2)
  - [x] ~~Message Queue Setup (RabbitMQ/Kafka/Redis)~~ — verworfen: Prefect Events + `duckdb.staging.pending_changes` erfüllen die Aufgabe ohne neuen Service; echte CDC ist auf DuckDB-über-Trino ohnehin nicht möglich
  - [x] FastAPI Webhook: `POST /events/new-data` in [app.py](../webservice/app.py) — publiziert per REST, ohne Prefect-SDK im Model-Image
  - [x] Change Detector: [change_detector.py](../prefect/change_detector.py) — diffed `row_count`/`column_count` gegen `duckdb.staging.source_watermarks`
  - [x] Event-getriggerte Deployments inkl. Cron-Fallback (alle 15 min) und Concurrency-Limits: [serve_flows.py](../prefect/serve_flows.py)
  - [x] Re-Prediction geänderter Tabellen: Dedup-Sperre in beiden Pipelines aufgehoben, Ergebnisse werden historisiert (append)
  - [x] Recovery verwaister Claims nach hartem Prozessabbruch
  - [x] **Bugfix 31.07. — `ICEBERG_COMMIT_ERROR` beim Claiming**: beide Pipelines hängen am selben Event, ihre Claim-`UPDATE`s trafen `staging.pending_changes` also gleichzeitig. Iceberg committet optimistisch und prüft beim Commit, ob ein neuerer Snapshot Dateien hinzugefügt hat, die auf das eigene Prädikat passen — der `keys`-Claim schreibt genau die Zeilen neu, auf die der `nf`-Claim filtert, also scheitert der zweite Commit. Die Konflikterkennung arbeitet auf **Datei-**, nicht auf Spaltenebene, die Trennung in `keys_*`/`nf_*`-Spalten schützt hier also nicht. Unter DuckDB fiel das nie auf. Fix: [`with_commit_retry`](../prefect/change_events.py) — Retry mit exponentiellem Backoff auf einer **neuen** Transaktion (eine gescheiterte ist unbrauchbar); alle Statements des Moduls sind unter Retry idempotent (`claim` filtert auf `IS NULL`, `complete`/`release` auf `claimed_by = run_id`). Verwendet in beiden Pipelines und im Detector, dessen Stale-Claim-Sweep dieselbe Tabelle schreibt
  - [x] `prefect`-Service ins `ml-services-monitoring`-Netz geholt und um den `serve()`-Runner erweitert (Server + Flow-Runner in einem Container)
  - [x] Prefect-Serverstate in Postgres statt SQLite: eigene DB `prefect` neben `mlflow_db`, angelegt von [ensure_database.py](../prefect/ensure_database.py) — überlebt jetzt `docker compose down`
  - [x] `prefect`-DB ins stündliche S3-Backup aufgenommen: eigener Service `prefect_postgres_backup` (Prefix `prefect_db_backups`, 7 Tage Retention) — das Image sichert nur je eine DB pro Container
  - [ ] **`prefect/Dockerfile` bauen** — der Service installiert `asyncpg`, `sqlalchemy`, `trino`, `pandas`, `python-dotenv`, `requests` beim Containerstart per `pip install`. Deshalb braucht der Healthcheck `start_period: 300s`, und jeder `docker compose up` zahlt die Installation erneut. Die Versionen sind hier ein zweites Mal gepinnt, unabhängig von `pyproject.toml` → Driftquelle
  - [ ] Event Logging zu Monitoring (Prometheus-Metriken für empfangene Events, Detection-Latenz, Backlog-Tiefe) — Prefect ist **kein** Scrape-Target in [prometheus.yaml](../prometheus/prometheus.yaml), die Streaming-Pipeline ist damit die einzige Komponente ohne Telemetrie
  - [ ] Housekeeping-Job: abgeschlossene `pending_changes`-Zeilen nach N Tagen löschen
  - [x] **Automatisierte Tests für Detector und Claim-Logik (30.07.)** — [test/test_pipelines/](../test/test_pipelines/), 70 Tests ohne Trino: `diff_snapshots` inkl. blindem Fleck, Claim-Protokoll gegen eine mitschreibende `FakeConnection`, `recompute_table_stats`. Details in 2.2
  - [ ] **Noch nie end-to-end gegen echte Trino-Daten gelaufen** — offen seit dem letzten Abgleich; der Demo-Pfad dafür ist in 4.1a beschrieben

---

### Phase 5: MLOps Tracking, Versioning, Retraining

- [~] **5.1** MLflow Tracking vollständig integrieren
  - [x] Experiment Tracking in **allen 5** Training-Scripts (`mlflow.start_run` je Kandidat)
  - [x] Hyperparameter-Suche mit `RandomizedSearchCV` + `StratifiedGroupKFold` in allen 5 Scripts (statt Optuna) — TODO „more than one model … with hyperparameter search" erledigt
  - [x] Metrics Logging (Accuracy, Precision, Recall, F1 — je train/test)
  - [~] Alle vier Kandidaten (RF, XGB, jeweils Baseline + RandomizedSearch) werden geloggt, bester per `f1_score` registriert
    - [x] `fk` und `cpk` (2026-08-05): Auswahl per `cv_f1_mean` aus 5-fold `StratifiedGroupKFold` statt aus **einem** Fold. Über die fünf Folds schwankt F1 um bis zu 0.21, die Auswahl lief also auf Rauschen. Zusätzlich geloggt: `cv_f1_std`, `cv_f1_per_fold`, gepoolte Out-of-Fold Confusion Matrix und PR-Kurve als PNG, plus `error_analysis_oof.json` (welche Spaltennamen werden verwechselt)
    - [ ] `pk`, `cfk`, `normalform` benutzen weiter den Einzel-Fold — gleiche Umstellung noch offen
  - [x] MLflow-Backend produktionsnah: Postgres als Backend-Store, MinIO/S3 als Artifact-Store, stündliches Postgres-Backup nach S3
  - [~] Artifacts: Confusion Matrix als `mlflow.log_dict` ✔ in allen 5 Scripts — **Feature Importance Plot fehlt** (in allen 5)
  - [ ] Dataset Tracking mit `mlflow.data` / `log_input` — würde zugleich die Frage „gegen welchen Datensatz wurde diese Version trainiert?" beantworten, die der Backtest in 6.5 implizit stellt

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

- [ ] **5.4** Retraining Path  *(offen — TODO „Automatic retraining?")* — **die größte verbleibende Lücke im MLOps-Kreislauf**
  - [ ] Manueller Retraining Trigger (API Endpoint `/model/retrain`)
  - [ ] Prefect Flow für Retraining (kapselt die `task_*_train_and_register.py`)
  - [ ] Retraining mit neuen Daten aus Trino
  - [ ] Auto-Register zu MLflow nach Retraining (Logik existiert bereits in den Scripts → wiederverwenden)
  - [x] Model Comparison (Old vs New) vor Alias-Umzug — `scripts/promote_model.py`, Vergleich per Test-F1
  - [x] Manuelle Promotion nach `prod` — `python scripts/promote_model.py --model <name>`, danach `docker compose restart model-service`
  - [ ] Trigger-Kriterium festlegen — **das Signal existiert jetzt**: die `evidently_*`-Gauges liefern Drift pro Track, und `model-quality-backtest` liefert stündlich F1 der *aktuell ausgelieferten* Version. Ein Prometheus-Alert auf einen F1-Einbruch oder einen anhaltend hohen `drifted_share` kann den Retraining-Flow anstoßen, der dann `scripts/promote_model.py` als Promotion-Gate nutzt
  - [ ] Baseline-Invalidierung mitdenken: nach einem Retraining beschreibt die Evidently-Referenz die *alte* Modellversion. [setup_stack.sh](../scripts/setup_stack.sh) löst das für den manuellen Weg (`--retrain` erzwingt `REBUILD_REFERENCE`); ein automatisierter Retraining-Flow muss dieselbe Kopplung herstellen, sonst misst das Monitoring danach den Versionswechsel statt der Daten

- [ ] **5.5** CI/CD für Modelle
  - [ ] GitHub Action für Model Training
  - [ ] Model Testing (Performance Thresholds, z.B. F1 ≥ Baseline)
  - [ ] Auto-Deploy zu Staging bei Success

---

### Phase 6: Monitoring

- [~] **6.1** Prometheus Service Monitoring
  - [x] `prometheus-fastapi-instrumentator` aktiviert, `/metrics` exponiert
  - [x] [prometheus/prometheus.yaml](../prometheus/prometheus.yaml) scraped `prometheus`, `model-service`, `evidently_service`
  - [x] Der Evidently-Service exportiert eigene Gauges über `prometheus_client` + `DispatcherMiddleware` auf `/metrics` — inkl. `evidently_reference_dataset_hash`, das [setup_stack.sh](../scripts/setup_stack.sh) als Beweis auswertet, dass der Service mit einer nutzbaren Baseline hochgekommen ist
  - [~] **Custom Metrics angelegt (30.07.)**: [webservice/metrics.py](../webservice/metrics.py) definiert vier `model_*`-Metriken plus die Helfer `record_success()` und `record_error()`. `prometheus_client` kommt transitiv über den Instrumentator, also ohne neue Abhängigkeit
    - [x] Error Counter — `model_prediction_errors_total{model,error_type}`, wird in allen fünf `except`-Blöcken über `record_error()` befüllt
    - [~] Prediction Counter — `model_predictions_total{model,predicted_class}` **deklariert, aber nie befüllt**
    - [~] Prediction Duration Histogram — `model_prediction_duration_seconds{model}` **deklariert, aber nie befüllt**
    - [~] Confidence Histogram — `model_prediction_probability{model}` **deklariert, aber nie befüllt**
    - [ ] Model Version Gauge (macht Rollbacks in Grafana sichtbar und ist Voraussetzung für 5.3) — noch nicht angelegt
  - [ ] 🐛 **`record_success()` wird nirgends aufgerufen.** [app.py](../webservice/app.py) importiert nur `record_error`; der Erfolgspfad der fünf Endpoints ruft keinen Recorder auf. Damit bleiben drei der vier Metriken dauerhaft ohne Datenpunkt. **Am laufenden Stack verifiziert:** `/metrics` liefert die vier `# HELP`-Zeilen (das Modul ist also geladen), aber `curl -s localhost:8080/metrics | grep -c '^model_'` gibt **0** zurück — auch unmittelbar nach einem erfolgreichen `POST /predict_pk`. Die Coverage-Lücke zeigte genau darauf: `metrics.py` 73 %, fehlende Zeilen 54–56 = der Rumpf von `record_success`. Folgen:
    - `ModelAlwaysPredictsOneClass` und `PredictionConfidenceCollapsed` (6.4) können **nie feuern** — ihre Zeitreihen existieren nicht
    - das Ziel „Model-Inferenz getrennt vom HTTP-Overhead" ist unerfüllt, obwohl das Histogram bereitsteht
    - Fix ist klein: Dauer um den `predict()`-Aufruf messen und `record_success(model, prediction_value, probability, duration)` vor dem `return` aufrufen — in allen fünf Handlern, bzw. einmal, wenn sie vorher wie in 3.1 vorgeschlagen zusammengefasst werden
  - [ ] Prefect-Flows als Scrape-Target oder Pushgateway anbinden (siehe 4.5)

- [~] **6.2** Golden Signals implementieren
  - [x] **Latency**: `http_request_duration_highr_seconds` (p95-Alert aktiv, Panel + Perzentil-Panel im Dashboard)
  - [x] **Traffic**: `http_requests_total` (No-Traffic-Alert aktiv, Rate je Endpoint im Dashboard)
  - [x] **Errors**: 4xx-/5xx-Rate-Alerts aktiv, Error-Rate-Panel im Dashboard
  - [~] **Saturation**: nur `process_resident_memory_bytes` — CPU und Disk fehlen (z.B. node-exporter/cAdvisor ergänzen)

- [~] **6.3** Grafana Dashboards — **weitgehend erledigt** (TODO „Graphana Dashboard" abgearbeitet)
  - [x] Grafana-Container in Compose inkl. Volume-Mounts, eigenem Branding (`grafana/img/`) und Admin-Credentials aus `.env` statt `admin/admin`
  - [x] Provisioning angelegt: [datasources.yaml](../grafana/provisioning/datasources/datasources.yaml) (Prometheus, `uid: prometheus`, default) und [grafana_dashboards.yaml](../grafana/provisioning/dashboards/grafana_dashboards.yaml) (File-Provider, `allowUiUpdates: false`)
  - [x] **5 Dashboards** in [grafana/dashboards/](../grafana/dashboards/):

    | Dashboard | UID | Inhalt |
    |---|---|---|
    | Model Service - Golden Signals | `model-service-golden-signals` | 10 Panels: Targets up, Request-Rate, 5xx-Rate, p95, Rate/Latenz je Endpoint, Perzentile, SLO „unter 500ms", Resident Memory |
    | Key Predict Data Drift | `evidently-key-drift` | Prediction-Drift-p-Value, drifted columns/share, Missing predictions, p-Value je Spalte — Modell-Dropdown (pk/cpk/fk/cfk) |
    | Key Model Performance | `evidently-model-quality` | F1/Precision/Recall/Log-Loss, jeweils current und current-vs-reference — Modell-Dropdown |
    | Normalform Predict Data Drift | `evidently-normalform-drift` | wie oben, ohne Dropdown, plus Macro-F1/Accuracy-Panels aus dem Backtest |
    | Normalform Model Performance | `evidently-normalform-quality` | Macro F1/Precision/Recall + Quality-Overview |

    Normalform hat ein eigenes Dashboard-Paar, weil das Modell multiclass ist und die Metriken macro-averaged sind — eine gemeinsame Variable mit den binären Modellen würde inkompatible Serien in dieselben Panels mischen
  - [ ] Dashboard für Data Pipeline (Queue-Länge, `pending_changes`-Tiefe, verarbeitete Zeilen, Detection-Latenz) — braucht zuerst die Prefect-Metriken aus 6.1
  - [ ] Dashboard für MLflow Experiments
  - [ ] Panel für Model Version / Prediction-Klassenverteilung — braucht die Custom Metrics aus 6.1

- [~] **6.4** Alerting
  - [x] [prometheus/alert.yaml](../prometheus/alert.yaml) mit **10 Regeln** in **3 Gruppen** (`service-health`, `golden-signals`, `model-health`):
    - [x] `HighServerErrorRate` (5xx >5%) / `HighClientErrorRate` (4xx >25%)
    - [x] `HighRequestLatencyP95` (>500ms)
    - [x] `NoTrafficReceived` + `InstanceDown` + `ModelServiceUnreachable`
    - [x] `HighMemoryUsage` (>1.5GB)
    - [x] **Gruppe `model-health` neu (30.07.)**: `ModelAlwaysPredictsOneClass` (30 min nur eine Klasse bei >50 Requests), `PredictionConfidenceCollapsed` (mittlere Confidence <0,6 über 15 min), `PredictionErrorsSpiking` (>0,1 Fehler/s)
  - [x] Alertmanager konfiguriert und an Prometheus angebunden ([alertmanager.yml](../alertmanager/alertmanager.yml), UI auf :9093)
  - [ ] ⚠️ **2 der 3 neuen Modell-Alerts können nie feuern**: `ModelAlwaysPredictsOneClass` braucht `model_predictions_total`, `PredictionConfidenceCollapsed` braucht `model_prediction_probability_sum`/`_count` — beide Serien existieren nicht, weil `record_success()` nicht aufgerufen wird (siehe 6.1). Nur `PredictionErrorsSpiking` ist funktionsfähig. Die Regeln sind korrekt formuliert, ihnen fehlt nur die Datenquelle
  - [ ] Notification Channels — Receiver `default` ist bewusst leer (null receiver); Slack/E-Mail via `*_file`-Secrets nachziehen
  - [ ] Einen Alert end-to-end testen und das Ergebnis dokumentieren *(bleibt das einzige „mindestens 1 funktionaler Alert"-Kriterium, das noch offen ist)*
  - [ ] Modell-Alerts auf die Evidently-Gauges legen (F1-Einbruch, anhaltender `drifted_share`) — liefert gleichzeitig das Trigger-Signal für 5.4

- [~] **6.5** Evidently Model Monitoring — **vom Tutorial-Stand auf produktiv umgebaut (28.07.2026)**
  - [x] Evidently-Service läuft als Container ([evidently_service/](../evidently_service/), Port 8085) und exponiert `/metrics`
  - [x] ~~Tutorial-Daten~~ **ersetzt**: keine `green_taxi_data` mehr. [config.yaml](../evidently_service/config.yaml) definiert **5 Tracks** (`pk_columns`, `cpk_columns`, `fk_columns`, `cfk_columns`, `nf_columns`), je mit eigener `DataDefinition`, Referenz, Report und rollierendem Fenster
  - [x] Reference Datasets aus den echten Features: [build_monitoring_references.py](../evidently_service/build_monitoring_references.py) leitet die Feature-Listen **aus den Pydantic-Request-Modellen** ab und scored die gelabelten Zeilen durch die live Endpoints. Damit gibt es keine dritte Stelle, an der ~35 Spaltennamen gepflegt werden müssten
  - [x] Kein `column_mapping` mehr nötig: numerisch/kategorial wird aus der Referenz-CSV inferiert (bool und Numerik mit ≤2 Werten → kategorial), `force_categorical` überschreibt Ausnahmen. Der Service loggt beim Start, welchen Split er inferiert hat
  - [x] Monitoring-Call in [app.py](../webservice/app.py) — nicht nur reaktiviert, sondern über [monitoring_client.py](../webservice/monitoring_client.py) + `BackgroundTasks` sauber gebaut: Post nach dem Response, `MONITORING_ENABLED`-Schalter, 2s-Timeout, **jede** Exception wird geschluckt (Log auf `debug`, damit ein toter Monitor nicht jede Prediction zumüllt)
  - [x] **Input Drift Detection** läuft aus dem Serving-Pfad (braucht keine Labels): `POST /iterate/<track>`, Fenster 60 Zeilen, Report alle 30 — bewusst nicht pro Request, sonst würde jede Prediction einen Vollreport über alle Spalten auslösen
  - [x] **Prediction Quality Monitoring** über [model_quality_backtest.py](../prefect/model_quality_backtest.py): F1/Precision/Recall brauchen Ground Truth, die es zur Predict-Zeit nicht gibt. Der Flow spielt stündlich die **Holdout**-Hälfte der gelabelten Daten durch die live Modelle (`/iterate_classification/<track>`) und beantwortet damit die engere, aber ehrliche Frage „scoren die gerade ausgelieferten Versionen noch wie ihre Baselines?"
  - [x] Stratifizierte Stichproben im Backtest, weil `composite_fk_target` nur ~2% positiv ist und ein einklassiges Fenster sonst eine irreführende Null exportieren würde — der Service überspringt solche Fenster bewusst
  - [x] Evidently-Metriken in Grafana visualisiert (4 der 5 Dashboards, siehe 6.3)
  - [x] Scheduled Reports mit Prefect: `model-quality-backtest/scheduled`, Cron `17 * * * *`
  - [x] Report-Introspektion per HTTP: `GET /tracks` (was wird überwacht), `GET /report/<track>`, `GET /classification_report/<track>`
  - [ ] **`MONITORED_TRACKS` in [setup_stack.sh](../scripts/setup_stack.sh) listet nur 4 Tracks** (`pk cpk fk cfk`). `build_monitoring_references.py` wird ohne `--only` aufgerufen und baut daher alle 5 — aber die Existenzprüfung und die Accuracy-Ausgabe übergehen `nf_columns`. Ein fehlender NF-Baseline-Build würde unbemerkt durchlaufen. Die Meldungen „all 4 predict endpoints" / „4 tracks" sind entsprechend falsch
  - [ ] Referenzen und Holdouts sind gitignored (`evidently_service/references/`, `data/holdouts/`) und müssen lokal erzeugt werden → in der README als Pflichtschritt dokumentieren, nicht nur im Setup-Skript
  - [ ] Log Loss für `nf_columns` bleibt bewusst weg (`/predict_normalform` gibt nur `max(predict_proba)` zurück, was für 4 Klassen nicht in eine Verteilung zurückgerechnet werden kann) → falls das Metrik-Loch stört: den Endpoint um die vollen Klassenwahrscheinlichkeiten erweitern
  - [ ] Das Drift-Fenster liegt im Prozessspeicher des Evidently-Service → ein Restart verwirft alle laufenden Fenster. Für Demos ausreichend, für Dauerbetrieb persistieren

- [ ] **6.6** Logging & Tracing
  - [ ] Structured Logging (JSON) — Webservice nutzt weiterhin `print()`/`traceback.print_exc()`; nur `monitoring_client.py` verwendet bereits `logging`
  - [ ] Log Aggregation (optional: Loki)
  - [ ] Correlation IDs für Request Tracing — mit vier ereignisgekoppelten Deployments wäre eine ID vom Webhook über den Detector bis in beide Pipelines inzwischen echter Mehrwert, nicht mehr nur Kosmetik

---

### Phase 7: Betrieb, Doku & offene Entscheidungen *(neu)*

- [~] **7.1** Developer Experience
  - [x] Start-up-Script: [scripts/setup_stack.sh](../scripts/setup_stack.sh) — inzwischen **7 Schritte**: Preflight, Compose mit Health-Waits, Training (idempotent), Registry- und Schema-Verifikation, Smoke-Test aller 5 Endpoints, **Evidently-Baselines inkl. Rebuild des Service**, optional Streaming via `--with-streaming`
  - [x] Skript auf Englisch übersetzt und Windows-tauglich gemacht (`3c50608`: kein `${PWD}` in Volume-Specs mehr, weil Git Bash es zu `C:/…` umschreibt und Compose die Doppelpunkte fehldeutet)
  - [x] Abschluss-Summary listet alle UIs, Dashboard-Pfade und die Monitoring-Besonderheiten (welche Metrik woher kommt, wie man den Backtest sofort auslöst)
  - [x] `.env.template` entfernt — [.env.example](../.env.example) ist die einzige Vorlage (`3404279`)
  - [x] README hat einen Quickstart (Environment → Train → Stack → Predict)
  - [ ] Zwei Abweichungen zwischen `.env.example` und dem tatsächlich gelesenen Environment: `.env.example` nennt `MONITORING_URL`, der Code liest `MONITORING_BASE_URL` ([monitoring_client.py](../webservice/monitoring_client.py), [model_quality_backtest.py](../prefect/model_quality_backtest.py)). Nicht genutzte Variablen aus der Vorlage streichen, genutzte ergänzen (`MONITORING_ENABLED`, `MONITORING_TIMEOUT_SECONDS`, `MODEL_API_URL`, `HOLDOUT_DIR`)
  - [ ] Die Schritt-Überschriften 4 und 7 im Setup-Skript sind noch deutsch/gemischt („Check Registry und Pydantic schema contract prüfen", „Streaming-Trigger")
  - [x] ~~`docker-compose.yaml` und `scripts/setup_stack.sh` uncommitted~~ — committet, Arbeitsverzeichnis ist sauber

- [ ] **7.2** Dokumentation aktuell halten — **🔴 der schlechteste Bereich im Projekt**
  - [ ] **Die README beschreibt Dateien und Zahlen, die es nicht gibt.** Belegte Abweichungen:

    | README sagt | Realität (30.07.) |
    |---|---|
    | `webservice/metrics.py` mit Model-Metriken | ✅ **existiert jetzt** — die README war der Umsetzung voraus, nicht falsch. Aber: 3 der 4 Metriken werden nie befüllt (6.1) |
    | „pytest with coverage" / `pytest --cov-report=term-missing` | ✅ **stimmt jetzt** — `pytest-cov` ist installiert, 273 Tests / 86 % |
    | `grafana/dashboards/golden-signals.json` | heißt `model_service_golden_signals.json` |
    | `evidently_service/build_reference.py` | heißt `build_monitoring_references.py` |
    | `test/test_api/` deckt die API-Oberfläche ab | heißt `test/test_webservice/`; die 75 API-Tests liegen dort |
    | „11 alert rules" | 10 Regeln in 3 Gruppen |
    | „Data pipeline · Prefect + dbt · 🔜 planned" | Prefect-Pipelines laufen inkl. 4 Deployments; dbt hat keine Models/Seeds |
    | „Retraining · `reload_models()` hook in place" | kein solcher Hook im Code |
    | „Model monitoring · input drift" | untertrieben: 5 Tracks, Drift **und** Classification Quality |

    Zwei der neun Punkte haben sich zwischen 29. und 30.07. von selbst erledigt — die README war
    dort der Umsetzung *voraus*, nicht hinterher. Die restlichen sieben bleiben: falsche Dateinamen
    und Zahlen, Fertiges als „geplant" markiert. Wer die README liest, sucht nach Dateien unter
    falschem Namen und unterschätzt, was schon läuft.
  - [ ] **7 Begleit-Dokumente wurden nie committet und liegen nicht mehr im Arbeitsverzeichnis**: `STREAMING_PIPELINE.md`, `STREAMING_PREDICTION_APPROACHES.md`, `FEATURE_PIPELINE_README.md`, `TRAIN_AND_PREDICT_FLOW.md`, `NORMALFORM_PREDICTION.md`, `PYDANTIC_PREFECT_DBT_FIX.md`, `INCREMENTAL_FEATURE_PROCESSING.md`. Sie existieren nur noch in einem Stash (`9fbb699`). `documentation/` enthält heute ausschließlich diese Datei und `flowchart.md` — alle Verweise darauf in früheren Planversionen waren tote Links. Entweder aus dem Stash zurückholen und committen oder die Inhalte in README/Modul-Docstrings überführen und die Dokumente als erledigt abhaken
  - [ ] [flowchart.md](flowchart.md) an den Ist-Stand anpassen (Postgres/MinIO-Backend, Prefect statt dbt-Python-Models, zwei Prediction-Pipelines, Evidently mit 5 Tracks, Backtest-Flow)
  - [ ] Monitoring-Architektur dokumentieren — die entscheidende Trennung „Drift ohne Labels aus dem Serving-Pfad **vs.** Classification Quality mit Labels aus dem Backtest" steht heute nur in Docstrings und in der Compose-/Setup-Ausgabe

- [ ] **7.3** Offene Entscheidungen
  - [ ] Anderer Katalog als DuckDB für Queue und `prediction_results`? (Persistenz/Concurrency bei parallelen Flows — mit zwei gleichzeitig getriggerten Pipelines jetzt akuter als vorher)
  - [ ] Wo werden Features final extrahiert — Prefect (Ist-Zustand) oder zurück nach dbt/SQL? Hängt an der dbt-Grundsatzentscheidung in 4.1
  - [ ] Klären: `min_value` / `max_value` werden nur zum Gruppieren und für Traceability genutzt, nie als Feature — korrekt so? (Der Backtest listet sie konsistent unter `IDENTITY_COLUMNS`, was dafür spricht.) Ergebnis festhalten
  - [~] Inkrementelle Feature-Verarbeitung — laut Projektnotizen mit dem Streaming-Trigger am 27.07. umgesetzt; History-Erhalt ist dabei Randbedingung. Ist-Stand gegen das ursprüngliche Design abgleichen und den Punkt schließen
  - [ ] Referenz-Datenhaltung: `evidently_service/references/` und `data/holdouts/` sind gitignored, aber die Baseline ist Teil der Monitoring-Aussage. Reproduzierbar aus dem Skript (heute) oder versioniert in MinIO (belastbarer)?

---

### Nächste 5 sinnvolle Schritte

1. **README auf den Ist-Stand bringen** (7.2) — sie ist das Einstiegsdokument und beschreibt derzeit ein anderes Projekt: nicht existierende Dateien, falsche Zahlen, Fertiges als „geplant" markiert. Billigste Korrektur mit dem größten Effekt.
2. **`dev` → `main` mergen und CI-Trigger geradeziehen** (2.3) — `main` steht auf dem Initial Commit, der Workflow horcht auf `develop`. Solange beides so ist, schützt kein Gate irgendetwas.
3. **Retraining-Flow in Prefect** (5.4) — der letzte fehlende Bogen im Kreislauf. Trainings-Scripts sind registrierungsfähig, das Trigger-Signal liefert jetzt der Quality-Backtest; nur Orchestrierung plus Baseline-Invalidierung fehlen.
4. **Custom Model-Metriken + Coverage** (6.1/2.1) — `pytest-cov` und `[dev]`-Extra nachziehen (reparieren zwei kaputte CI-Steps), Prediction-Counter/Duration/Version-Gauge ergänzen (schalten die letzten Grafana-Panels frei).
5. **Docker-Build-Workflow + Dockerfile-Härtung** (2.4/3.3) — Multi-stage, non-root, `.dockerignore`, `HEALTHCHECK`, Push zu GHCR. Für `evidently_service` zusätzlich `references/` mounten statt einbacken; das streicht einen Rebuild-Schritt aus dem Setup.

**Danach, mit klarem Vorlauf:** dbt-Grundsatzentscheidung (4.1), `prefect/Dockerfile` (4.5), Streaming end-to-end gegen echte Trino-Daten (4.5), Alert-Receiver + ein End-to-End-Test (6.4).

---

## 3. ARBEITSPAKETE MIT TEILSCHRITTEN

> **Hinweis:** Die Abschnitte 3 bis 9 sind der *ursprüngliche* Plan vom Projektstart und
> werden absichtlich unverändert gelassen — sie dokumentieren die Ausgangsannahmen.
> Der verbindliche Ist-Stand steht in **Abschnitt 2**. Wo beides auseinandergeht, gilt
> Abschnitt 2. Die größten bewussten Abweichungen:
>
> - **Arbeitspaket 6** plant RabbitMQ/Redis; umgesetzt wurden Prefect Events + Watermark-Diff
>   ohne zusätzlichen Service (echte CDC ist auf DuckDB-über-Trino nicht möglich).
> - **Arbeitspaket 4** plant dbt-Models; die Feature-Extraktion liegt in Prefect, weil
>   dbt-trino keine Python-Models unterstützt.
> - **Arbeitspaket 11** plant Evidently mit *einem* Referenzdatensatz; umgesetzt sind 5 Tracks
>   mit getrennten Drift- und Classification-Quality-Pfaden.
> - MLflow **Stages** (Arbeitspaket 7/8) sind in MLflow 3.x deprecated → das Projekt nutzt **Aliase**.

**Abgleich Arbeitspakete 1–5 gegen den Code (30.07.2026)** — Details jeweils im „Ist-Stand"-Block
unter dem Arbeitspaket. 9 von 15 Deliverables erfüllt.

| AP | Deliverable 1 | Deliverable 2 | Deliverable 3 | Fazit |
|----|---------------|---------------|---------------|-------|
| **1** Foundation | Ruff fehlerfrei ✅ | ≥10 Tests / >50% Cov ✅ **273 / 86%** | CI auf jedem Push 🟡 | 🟢 **erledigt**, Testziel weit übertroffen |
| **2** Docker & Registry | Image <500MB ❌ **1,65 GB** | GHCR-Push ❌ | Compose auf GHCR ❌ | 🔴 **nicht begonnen** |
| **3** API Enhancement | OpenAPI vollständig ❌ | Alle Endpoints getestet ✅ | Health Checks ❌ | 🟡 Modelle + Tests fertig, Endpoint-Liste offen |
| **4** Data Pipeline | dbt generiert Test-Daten ❌ | Prefect Server ✅ | Flow orchestriert dbt ❌ *(verworfen)* | 🟡 Prefect-Hälfte fertig, dbt zurückgebaut |
| **5** Batch Pipeline | Training Flow automatisch ❌ | Batch Prediction per CSV ❌ | Flows schedulbar ✅ | 🟡 Scheduling steht, Training + CSV fehlen |

Drei Muster fallen dabei auf:

1. **Testing ist von der Schlusslicht- zur Vorzeigedisziplin geworden** — AP1 war beim letzten
   Abgleich die größte Lücke und ist heute mit 273 Tests und 86 % Coverage übererfüllt.
2. **Was orchestriert werden musste, läuft; was Training betrifft, nicht.** Prediction-Pipelines,
   Event-Kette, Scheduling und Monitoring stehen — die fünf Trainings-Skripte sind die einzigen
   Komponenten, die Prefect nie erreicht haben (`grep -l prefect task_*/*.py` ist leer). Das ist
   dieselbe Lücke wie 5.4 und sollte einmal gebaut werden, nicht zweimal.
3. **Zwei Deliverables wurden bewusst anders gelöst als geplant** (dbt-Models → Prefect,
   Message Queue → Prefect Events) und sind keine Schulden. Zwei weitere sind es sehr wohl:
   Docker-Härtung (AP2) und der CSV-Eingang (AP5).

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

> #### 🟢 Ist-Stand AP1 (30.07.2026): erledigt, Testziel deutlich übertroffen
>
> | Deliverable | Ist | Nachweis |
> |---|---|---|
> | Ruff läuft ohne Fehler | ✅ | `ruff check` + `ruff format --check` grün in CI **und** als pre-commit-Hook |
> | ≥10 Unit Tests, >50% Coverage | ✅ **273 Tests, 86,38%** | `pytest --cov` → „273 passed in 7.06s", `fail_under = 80` erreicht |
> | CI läuft auf jedem Push | 🟡 | Feature-Branches ja, `dev` selbst nein (Trigger horcht auf `develop`) |
>
> **Abweichungen von der Planung — alle bewusst:**
> - Ruff-Konfiguration liegt in [ruff.toml](../ruff.toml), nicht in `pyproject.toml [tool.ruff]`. Inhaltlich erfüllt sie den Plan und geht darüber hinaus: `line-length = 100` ✓, `target-version = "py311"` ✓, `select` enthält die geplanten `E, F, I, N, W, B, Q` **plus** `C90`, `UP`, `ANN` u.a.
> - Testverzeichnis heißt `test/` statt `tests/` und hat **vier** Pakete statt der geplanten drei: `test_data/`, `test_models/`, `test_pipelines/`, `test_webservice/`. Die Rolle des geplanten `test_api/` erfüllt [test_webservice/test_api_endpoints.py](../test/test_webservice/test_api_endpoints.py) (75 Tests).
> - `pytest-asyncio` wurde **nicht** installiert und ist nicht nötig — die Endpoints sind synchron, `TestClient` reicht. Stattdessen `httpx2` als `TestClient`-Backend.
> - Kein `dev`-Extra: das Test-Tooling steht in `dependencies` (Begründung in 2.1).
>
> **Restpunkte** (in 2.3 geführt): Coverage-Badge fehlt; CI ruft `pytest` **ohne** `--cov`, weshalb die beiden Upload-Steps weiterhin nicht existierende `coverage.xml`/`htmlcov/` hochladen — die Coverage-Konfiguration ist da, CI nutzt sie nur nicht.

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

> #### 🔴 Ist-Stand AP2 (30.07.2026): nichts davon umgesetzt
>
> | Deliverable | Ist | Nachweis |
> |---|---|---|
> | Docker Image <500MB | ❌ **1,65 GB** | `docker images` → `model-service:latest` 1,65 GB, `evidently_service:latest` 1,2 GB. Faktor **3,3** über dem Ziel |
> | Automatischer Push zu GHCR | ❌ | `.github/workflows/` enthält nur `ci.yml` |
> | Compose nutzt GHCR Images | ❌ | beide Services haben `build:`-Blöcke, keine `image:`-Referenzen |
>
> **Nicht umgesetzte Teilschritte:** Multi-stage Build, non-root User, `HEALTHCHECK`, `.dockerignore`
> (existiert in **keinem** der drei möglichen Pfade), Image-Tagging, GHCR-Secret.
>
> **Zwei Beobachtungen für die Umsetzung:**
> - In [webservice/Dockerfile](../webservice/Dockerfile) steht `COPY . /app` **vor** `pip install`. Jede Code-Änderung invalidiert damit die komplette Dependency-Installation — der Build ist unnötig teuer, und ohne `.dockerignore` wandert `__pycache__/` mit ins Image. Beides trägt direkt zu den 1,65 GB bei.
> - Das `CMD` im Beispiel-Snippet oben (`uvicorn webservice.app:app`) passt **nicht** zum Projekt: der Build-Kontext ist `webservice/` selbst, `WORKDIR` ist `/app`, und die Module importieren sich flach (`from predict import predict`). Korrekt ist `app:app`. Beim Übernehmen des Snippets nicht mitkopieren.
>
> Bleibt Priorität 5 der „Nächsten 5 Schritte" — nach der Doku, dem Merge-Gate und dem Retraining-Pfad.

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

> #### 🟡 Ist-Stand AP3 (30.07.2026): Modelle und Tests fertig, die Endpoint-Liste nicht
>
> | Deliverable | Ist | Nachweis |
> |---|---|---|
> | OpenAPI Docs vollständig | ❌ | nur [data_model_events.py](../webservice/data_model_events.py) nutzt `Field(description=…)` und `json_schema_extra.examples`. Die **fünf Predict-Modelle haben keine Beschreibungen und keine Examples** — `/docs` zeigt dort nur nackte Feldnamen |
> | Alle Endpoints mit Tests | ✅ | [test_api_endpoints.py](../test/test_webservice/test_api_endpoints.py), 75 Tests: alle 5 Routen ×(200, Echo, Modellname, Track, Feature-Reihenfolge, 400, 422), dazu `GET /`, `/metrics`, `POST /events/new-data` (202/503) |
> | Health Checks funktionieren | ❌ | es gibt nur `GET /` mit statischer Message |
>
> **Endpoint-Liste des Plans vs. Code** — keiner der sechs geplanten Pfade existiert:
>
> | Geplant | Im Code |
> |---|---|
> | `/api/v1/predict/pk` · `/fk` · `/normalization` | `/predict_pk`, `/predict_cpk`, `/predict_fk`, `/predict_cfk`, `/predict_normalform` — flach, ohne Version |
> | `/api/v1/health/live` · `/health/ready` | ❌ nicht vorhanden |
> | `/api/v1/model/info` | ❌ nicht vorhanden |
>
> Zusätzlich existiert `POST /events/new-data`, das im Plan nicht vorgesehen war (AP6).
>
> **Teilschritt „Error Handling":** 🟡 einheitlich `HTTPException` (400 bei Predict-/Input-Fehlern,
> 503 beim Event-Publish) und seit dem 30.07. `record_error(<model>, error)` in jedem der fünf
> `except`-Blöcke. Es gibt aber **keinen** `@app.exception_handler` und kein Fehler-Response-Modell —
> Fehler sind ein `detail`-String, keine strukturierte Antwort.
>
> **Ein Health-Check hätte hier konkreten Nutzen** und ist nicht nur Formalität: `model-service` ist
> der einzige Compose-Service ohne `healthcheck`, und die Modelle werden lazy beim ersten Request
> geladen (3.2). Ein `/health/ready`, das die MLflow-Erreichbarkeit prüft, würde beides sichtbar machen
> und `depends_on: service_healthy` auf den Service ermöglichen.

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

> #### 🟡 Ist-Stand AP4 (30.07.2026): Prefect-Hälfte fertig, dbt-Hälfte zurückgebaut
>
> | Deliverable | Ist | Nachweis |
> |---|---|---|
> | dbt läuft und generiert Test-Daten | ❌ | von dbt sind nur `dbt_project.yml`, `profiles.yml`, `.user.yml` getrackt. `models/` leer, `seeds/` **entfernt** (`ad3ccdf`), keine Tests |
> | Prefect Server läuft lokal | ✅ | Service `prefect` in Compose auf :4200, Postgres-Backend, `healthy` |
> | Erster Flow orchestriert dbt | ❌ **bewusst verworfen** | dbt-trino unterstützt keine Python-Models; die Feature-Extraktion liegt in [pk_fk_pipeline.py](../prefect/pk_fk_pipeline.py) |
>
> **Abweichungen:** `dbt-trino` statt `dbt-duckdb` (DuckDB wird als Trino-Katalog angesprochen);
> Prefect läuft nicht via `prefect server start` von Hand, sondern als Compose-Service mit
> `serve()`-Runner im selben Container.
>
> **Der dritte Teilschritt „Test Data Generation" ist offen:** keine dbt-Seeds mehr, kein
> Generator-Skript. Teilweise abgedeckt durch
> [build_monitoring_references.py](../evidently_service/build_monitoring_references.py), das den
> gelabelten Datensatz stratifiziert in Referenz + Holdout splittet — das ist aber
> **Monitoring**-Datenerzeugung, kein Ersatz für synthetische Testtabellen mit bekannten
> PK/FK/Normalform-Eigenschaften. Alles hängt weiter am eingefrorenen, handgelabelten Datensatz.
>
> ⚠️ **Grundsatzentscheidung fällig (siehe 4.1):** dbt ist derzeit Konfiguration ohne Wirkung —
> kein Aufruf aus Prefect, Compose oder `setup_stack.sh`. Entweder entlang
> der Demo-Idee reaktivieren (tpch → `new_predict_data` → triggert die Streaming-Pipeline, siehe
> 4.1a) oder `dbt-core`/`dbt-trino`/`prefect-dbt` aus `pyproject.toml` streichen.

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

> #### 🟡 Ist-Stand AP5 (30.07.2026): Scheduling steht, Training und CSV-Eingang fehlen
>
> | Deliverable | Ist | Nachweis |
> |---|---|---|
> | Training Flow läuft automatisch | ❌ | **kein** Trainings-Skript importiert Prefect (`grep -l prefect task_*/*.py` → leer). Die fünf `task_*_train_and_register.py` laufen standalone |
> | Batch Prediction akzeptiert CSV | ❌ | `grep read_csv prefect/*.py` findet **nur** [model_quality_backtest.py:75](../prefect/model_quality_backtest.py#L75) — und das liest Holdout-Labels fürs Monitoring, ist keine Batch-Prediction. Beide Prediction-Flows nehmen nur `target_schemas` / `use_pending_changes` und lesen aus Trino |
> | Flows sind schedulbar | ✅ | 4 Deployments in [serve_flows.py](../prefect/serve_flows.py), Cron `*/15 * * * *` und `17 * * * *`, dazu Event-Trigger und Concurrency-Limits |
>
> **Teilschritt 1 (Batch Training Flow)** ist damit komplett offen: keine Flow-Definition, keine
> Integration des Trainings-Codes, kein orchestriertes MLflow-Logging, kein Error Handling.
> Wichtig: das Training *loggt* bereits vollständig nach MLflow und registriert das beste Modell —
> es fehlt ausschließlich die Orchestrierung. Deckungsgleich mit **5.4** (Retraining), das denselben
> Flow braucht; beide sollten in einem Zug gebaut werden, nicht zweimal.
>
> **Teilschritt 2 (Batch Prediction Flow)** ist zur Hälfte da, aber anders als geplant: Batch-Prediction
> über **Trino** läuft (Feature-Extraktion aus `information_schema`, Prediction-Queue,
> `fetch_new_rows`/`predict_batch`, Ergebnisse nach `duckdb.prediction_results.key_results` /
> `nf_results`). Was fehlt, ist der **CSV-Eingang samt Input-Validation** — der Plan sah CSV als
> Quelle, umgesetzt wurde ein Datenbank-Pfad.
>
> **Teilschritt 3 (Scheduling)** ist vollständig und geht über den Plan hinaus: neben Cron auch
> Event-Trigger, Concurrency-Limits mit bewusst gewählten Kollisionsstrategien und ein Cron-Fallback
> als Sicherheitsnetz (Details in 4.3).

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

*Ist-Stand, Stand 29. Juli 2026 — abweichend vom ursprünglichen Plan, wo vermerkt.*

| Kategorie | Technologie | Zweck | Status |
|-----------|-------------|-------|--------|
| **Code Quality** | Ruff `0.16.0` + pre-commit + gitleaks | Linting, Formatting, Secret-Scan | ✅ |
| **Testing** | pytest (56 Tests) | Datenqualität + Schema-Contract | 🟡 `pytest-cov`/`pytest-mock` fehlen |
| **CI/CD** | GitHub Actions | Lint + Test | 🟡 Trigger-Branch falsch, Coverage-Steps ins Leere |
| **Container** | Docker, Docker Compose (11 Services) | Containerization | 🟡 nicht gehärtet, kein `.dockerignore` |
| **Registry** | GitHub Container Registry | Image Storage | ❌ offen, Compose baut lokal |
| **API** | FastAPI, Pydantic | Model Service, 5 typisierte Endpoints + Event-Webhook | ✅ |
| **Orchestration** | Prefect 3 | 4 Deployments: Detector, 2 Prediction-Pipelines, Quality-Backtest | ✅ |
| **Data Transform** | ~~dbt~~ | war für Feature-Extraktion geplant | ❌ entkernt, Entscheidung offen (4.1) |
| **Messaging** | ~~RabbitMQ/Redis~~ → **Prefect Events** | Event-Transport ohne zusätzlichen Service | ✅ bewusste Abweichung |
| **ML Tracking** | MLflow 3 + Postgres + MinIO/S3 | Experiment Tracking & Registry (Aliase statt Stages) | ✅ |
| **Service Monitoring** | Prometheus | Golden Signals, 7 Alert-Regeln | 🟡 Custom Model-Metriken fehlen |
| **Alerting** | Alertmanager | Routing, Dedup | 🟡 Receiver ist null |
| **Visualization** | Grafana 13 | 5 provisionierte Dashboards | 🟢 Pipeline-/MLflow-Dashboard fehlt |
| **Model Monitoring** | Evidently | 5 Tracks: Drift (serving) + Classification Quality (Backtest) | ✅ |
| **Storage** | Trino/DuckDB, Postgres, MinIO | Quelldaten & Ergebnisse, State, Artefakte | ✅ |
| **Backup** | postgres-backup-s3 (2 Instanzen) | `mlflow_db` (30 T) + `prefect` (7 T) nach MinIO | ✅ |
| **Logging** | `print()` | Application Logs | ❌ Structured Logging offen (6.6) |

---

## 6. ERFOLGSKRITERIEN

*Abgleich zum 29. Juli 2026 — ✅ erfüllt · 🟡 teilweise · ❌ offen.*

### Code Quality

- ✅ Ruff läuft ohne Errors (Lint + Format, in CI **und** pre-commit)
- ✅ Type Hints in Training-Scripts, Prefect-Flows und Webservice
- ❌ >70% Test Coverage — Coverage wird gar nicht gemessen (`pytest-cov` fehlt)

### CI/CD

- 🟡 Alle Commits werden getestet — Feature-Branches ja, `dev` selbst nicht (Trigger horcht auf `develop`)
- ❌ Docker Images werden automatisch gebaut — kein `docker-build.yml`, kein GHCR-Push
- ❌ Tests müssen grün sein vor Merge — `main` steht auf dem Initial Commit, PRs laufen nach `dev` ohne CI

### Data Pipeline

- ❌ Batch Training läuft automatisch — die 5 Trainings-Scripts laufen standalone
- 🟡 Streaming Pipeline verarbeitet Events — Event-Kette und Deployments stehen, aber noch nie gegen echte Trino-Daten gelaufen
- ❌ dbt generiert Test-Daten — Seeds wurden entfernt, dbt ist ohne Models/Seeds

### MLOps

- ✅ Alle Experimente in MLflow geloggt (5 Modelle × 4 Kandidaten, Metriken/Signatur/Input-Example)
- ✅ Produktions-Models im Registry, Deploy über Alias `@dev`
- ❌ Retraining manuell auslösbar — kein Endpoint, kein Flow

### Monitoring

- 🟡 Alle 4 Golden Signals messbar — Latency/Traffic/Errors ✅, Saturation nur Memory
- 🟡 Mindestens 1 Alert konfiguriert — 7 Regeln feuern, aber der Receiver ist null und nichts ist end-to-end getestet
- ✅ Evidently trackt Drift — 5 Tracks aus dem Serving-Pfad, dazu Classification Quality aus dem stündlichen Backtest, alles in Grafana sichtbar

**Fazit:** Der Monitoring- und Orchestrierungsteil ist am weitesten, der **Kreislauf ist noch nicht
geschlossen** (kein Retraining), und die **Nachweisschicht** — Coverage, Merge-Gate, Doku — ist die
schwächste Stelle.

---

## 7. NÄCHSTE SCHRITTE

Die verbindliche, priorisierte Liste steht in **Abschnitt 2 → „Nächste 5 sinnvolle Schritte"**.
Kurzfassung der Reihenfolge, in der die Arbeitspakete real abgeschlossen werden sollten:

1. **Nachziehen, was schon läuft**: README/Doku auf den Ist-Stand (7.2), `dev` → `main` + CI-Trigger (2.3)
2. **Kreislauf schließen**: Retraining-Flow inkl. Baseline-Invalidierung (5.4, Arbeitspaket 8)
3. **Messbar machen**: Coverage + Custom Model-Metriken (2.1, 6.1)
4. **Produktionsreife**: Docker-Härtung + GHCR (2.4/3.3, Arbeitspaket 2)
5. **Grundsatzfragen**: Rolle von dbt (4.1), `prefect/Dockerfile` (4.5), Alert-Receiver (6.4)

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
# UI (läuft im Stack als Service auf :5000)
mlflow ui --host 0.0.0.0 --port 5000

# List registered models
mlflow models list
```

> **Achtung:** `mlflow models transition-stage --stage Production` aus dem Ursprungsplan gilt
> nicht mehr — Stages sind in MLflow 3.x deprecated. Das Projekt nutzt **Aliase**
> (`set_registered_model_alias(name, "dev", version)`), und der Webservice lädt
> `models:/<name>@dev`.

### Projekt-spezifisch

```bash
# Kompletter Stack von null auf lauffähig (7 Schritte, idempotent)
./scripts/setup_stack.sh
./scripts/setup_stack.sh --retrain              # Modelle neu trainieren + Baseline neu bauen
./scripts/setup_stack.sh --with-streaming       # zusätzlich den Streaming-Trigger auslösen

# Was wird überwacht?
curl localhost:8085/tracks

# Drift-Report als HTML
open http://localhost:8085/report/pk_columns

# Model-Quality-Backtest sofort laufen lassen (statt auf Minute :17 zu warten)
docker compose exec -T -w /opt/flows prefect python model_quality_backtest.py

# Evidently-Baselines neu erzeugen (nach Retraining oder Feature-Änderung Pflicht)
python evidently_service/build_monitoring_references.py --model-url http://localhost:8080
docker compose up -d --build evidently_service   # COPY . /app -> Rebuild nötig

# Einzel-Prediction und Streaming-Trigger
bash curl_tests/test_curl_predict_pk.sh
bash trigger_prefect_pipeline.sh
```

### Pre-commit

```bash
pre-commit install            # einmalig, Hooks aktivieren
pre-commit run --all-files    # alles prüfen, ohne zu committen
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
