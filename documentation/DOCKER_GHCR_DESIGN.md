# Design: Docker-Build über GitHub Actions + Image-Härtung

**Stand: 12. August 2026** · Betrifft [MLOPS_PLAN.md](MLOPS_PLAN.md) **2.4** (Docker Build &
GHCR Pipeline), **3.3** (Dockerfile-Optimierung), **Arbeitspaket 2** (Docker & Registry) und
den Trigger-Teil von **2.3**.

Dieses Dokument ist der verbindliche Entwurf. Die Umsetzung ist in fünf Pull Requests
geschnitten (Abschnitt 4); jeder hat ein eigenes Verifikationsgate und einen eigenen
Rollback. Was bewusst *nicht* gemacht wird, steht in Abschnitt 7 — mit Begründung, damit
die Punkte nicht beim nächsten Abgleich als vergessen gelten.

---

## 1. Ausgangslage (gemessen, nicht angenommen)

| Beobachtung | Wert | Quelle |
|---|---|---|
| Buildbare Images | **3**, nicht 2 | `webservice/`, `evidently_service/`, `prefect/` haben je ein Dockerfile |
| `model-service` unkomprimiert | **1,51 GB** | `docker image inspect --format '{{.Size}}'` |
| davon in *einer* Layer | **1,49 GB** | `docker history` → die `pip install -r requirements.txt`-Zeile |
| `apt`/`libgomp1`-Layer | 1,49 MB | dito |
| Anwendungscode | 54,7 kB | dito (`COPY . /app`) |
| `evidently_service` | 963 MB | `docker images` |
| `prefect` | 1,02 GB | `docker images` |
| Basis-Image `site-packages` | **19 MB** (`pip 24.0`, `setuptools 65.5.1`, `wheel 0.45.1`) | `docker run --rm python:3.11.13-slim-bookworm pip list` |
| Repository | **private**, Default-Branch `main` | `gh repo view --json visibility,defaultBranchRef` |
| `origin/main` | `e203865 Initial commit`, **161 Commits** hinter `dev` | `git rev-list --count origin/main..origin/dev` |

Zwei Punkte in 2.4/3.3 sind seit dem letzten Abgleich **schon erledigt** und dürfen nicht
erneut eingeplant werden:

- `.dockerignore` existiert in allen drei Kontexten (3.3 sagt „in keinem").
  `prefect/.dockerignore` ist allerdings dünn — nur `__pycache__/` und `*.pyc`, es fehlt
  der `.env`-Ausschluss, den die anderen zwei haben.
- `webservice/Dockerfile` kopiert `requirements.txt` **vor** `pip install`. Die in 3.3 und
  AP2 beschriebene Cache-Invalidierung gilt heute nur noch für
  `evidently_service/Dockerfile`.

Ebenfalls überholt: **es braucht kein `GHCR_TOKEN`-Secret.** Der eingebaute
`GITHUB_TOKEN` kann mit `permissions: packages: write` in den Namespace des
Repository-Owners pushen. Der Punkt in 2.4 entfällt und wird beim Abhaken als *entfallen*
markiert, nicht als *erledigt*.

### Was die Zahlen für den Plan bedeuten

**Das Volumen liegt zu 100 % in den Python-Abhängigkeiten.** Multi-stage Build, non-root
User und `.dockerignore` sind Sicherheits- und Cache-Maßnahmen — sie bringen zusammen
geschätzt 20–60 MB, nicht mehrere hundert. Das AP2-Ziel „<500 MB" hängt allein daran, ob
`mlflow` durch `mlflow-skinny` ersetzt werden kann (PR3). Diese Erwartung wird hier
explizit festgehalten, damit die Härtung nicht später als gescheiterte Volumenmaßnahme
gelesen wird.

**Daraus folgt eine Reihenfolge:** Erst Härtung + Workflow, dann Slimming, *dann* Compose
auf Pull umstellen. Umgekehrt würde jeder Entwickler pro `pull` ~600 MB komprimierte
Layer ziehen, bevor überhaupt gemessen ist, ob das nötig ist.

---

## 2. Entschiedene Fragen

| Frage | Entscheidung | Begründung |
|---|---|---|
| Trigger | `push` auf `dev` + `main` + Tag `v*`; `pull_request` gegen `dev`/`main` | siehe unten |
| Manueller Lauf | `workflow_dispatch` | **Nachtrag 12.08.** Zweck ist *Neuveröffentlichung ohne Code-Änderung*: alle drei Dockerfiles führen `apt-get upgrade` zur Buildzeit aus, ein Rebuild zieht also frische Debian-Security-Patches — anders nur durch einen Commit erreichbar. ⚠️ GitHub bietet das Event erst an, wenn die Workflow-Datei auf dem **Default-Branch** liegt; das ist `main` und steht auf dem Initial Commit. Der „Run workflow"-Knopf erscheint also erst nach dem `dev`→`main`-Merge. Damit ist er **kein** Weg, den Publish-Pfad vor dem Merge zu verifizieren |
| Push bei PRs | **nein**, nur bauen | Ein `GITHUB_TOKEN` aus einem Fork-PR ist read-only — ein Push *müsste* scheitern. „PR baut, Merge pusht" ist damit nicht defensiv, sondern die einzige funktionierende Form |
| `latest` zeigt auf | **`dev`** (dokumentierte Abweichung) | `main` steht auf dem Initial Commit. Ein `latest` von `main` wäre entweder nicht existent oder würde den Initial Commit ausliefern. Eine kommentierte Zeile im Workflow schaltet es nach dem `dev`→`main`-Merge um |
| `dev`→`main`-Merge | **nicht Teil dieser Arbeit** | 161 Commits, fremdes Repository, hängt an Branch Protection → gehört zu 2.3 |
| Paket-Namen | flach: `ghcr.io/chrdro/alligator-{model-service,evidently-service,prefect}` | kurz genug für Compose und lesbar in der Demo. Alternative wäre verschachtelt (`ghcr.io/chrdro/mlops-alligator-metaswamp/model-service`), was automatisch ans Repo bindet — für drei Images in einem persönlichen Namespace kein ausreichender Vorteil |
| Paket-Sichtbarkeit | **private** (erbt vom Repo), Neubewertung in PR5 | Das Kontingent ist kein Argument (siehe unten). Öffentlich bedeutet: der komplette in das Image gebackene Quellcode von `webservice/`, `evidently_service/` und `prefect/` wird öffentlich, obwohl das Repository privat ist. Diese Entscheidung wird dann getroffen, wenn mit PR3 die echten Volumina vorliegen |
| Architekturen | **nur `linux/amd64`** | arm64 ginge nur über QEMU-Emulation. `scipy`/`lightgbm`/`xgboost` in dieser Größenordnung emuliert zu installieren treibt den Build von ~3 min auf 20+ min. Deployment-Ziel sind x86-Server und Docker Desktop. Der Plan führt Multi-arch selbst als „(optional)" |
| Prefect non-root | **ja, aber als letzter PR** | Es ist der einzige Service mit beschreibbarem Bind-Mount und eigenem State-Verzeichnis (`PREFECT_HOME`). Isoliert am Ende heißt: wenn es sich wehrt, fällt genau dieser Schritt weg und nicht die Härtung der anderen zwei |
| Trivy-Ergebnis | **Step-Summary + JSON-Artifact**, kein SARIF-Upload | Code Scanning braucht für *private* Repositories GitHub Advanced Security. `github/codeql-action/upload-sarif` scheitert hier mit „Advanced Security must be enabled". Das JSON-Artifact ist ohnehin das bessere Ziel: es ist genau das Eingabeformat von [`scripts/generate_vulnerability_report.py`](../scripts/generate_vulnerability_report.py) |

### Kontingent — geprüft und entkräftet

Die naheliegende Sorge bei 3,6 GB Images in einem privaten Repository ist das
GitHub-Packages-Kontingent (Free: 500 MB Storage / 1 GB Transfer). Sie greift nicht:
laut GitHubs Billing-Dokumentation ist *„Container image storage and bandwidth for the
Container registry currently free"* — die Container Registry wird getrennt von diesem
Kontingent geführt. Konsequenzen für den Entwurf:

- Die Sichtbarkeitsentscheidung ist frei von Kostendruck (siehe Tabelle oben).
- Trotzdem sinnvoll: ein Aufräum-Workflow (PR5), weil pro Push eine `sha-`-Version
  entsteht und die Paketseite sonst nach wenigen Wochen unlesbar ist.
- Bekannter Vorbehalt: GHCR ist in einem „soft billing"-Zustand. GitHub kündigt
  Änderungen mindestens einen Monat vorher an; sehr hoher Egress kann unabhängig davon
  auffallen. Für dieses Projekt irrelevant, hier nur festgehalten, damit die Annahme
  datiert ist.

---

## 3. Zielbild

```
┌──────────────────────────────────────────────────────────────────────────┐
│  push dev / main / v*                    pull_request → dev / main       │
└───────────────┬──────────────────────────────────┬───────────────────────┘
                │                                  │
        ┌───────▼──────────────────────────────────▼───────┐
        │  docker-build.yml   (matrix: 3 Images)           │
        │  setup-buildx → [login]* → metadata → build      │
        │  cache: type=gha, scope je Image                 │
        │  * login/push nur wenn event != pull_request     │
        └───────┬──────────────────────────────────────────┘
                │
        ┌───────▼───────────┐        ┌──────────────────────────────┐
        │  Trivy je Image   │───────▶│  $GITHUB_STEP_SUMMARY (Tab.) │
        │  CRITICAL,HIGH    │        │  + JSON-Artifact (30 Tage)   │
        │  exit-code 0      │        └──────────────┬───────────────┘
        └───────┬───────────┘                       │
                │ push (kein PR)                    ▼
        ┌───────▼──────────────────────┐   scripts/generate_
        │  ghcr.io/chrdro/alligator-*  │   vulnerability_report.py
        │  :dev :latest :sha-xxxxxxx   │   → dieselbe PDF wie lokal
        │  :main  :1.0.0 :1.0          │
        └───────┬──────────────────────┘
                │
    ┌───────────┴────────────┬─────────────────────────────────┐
    │                        │                                 │
docker-compose.yaml     docker-compose.ghcr.yaml         setup_stack.sh
image: + build:         pull_policy: always              unverändert
(lokal bauen,           (aus GHCR ziehen)                (--build)
 Default-Pfad)
```

Zwei Pfade, kein Umschalten: lokal wird gebaut, für Deployment/Fremdrechner wird gezogen.
`setup_stack.sh` bleibt davon unberührt.

---

## 4. Umsetzung in fünf PRs

Reihenfolge ist eine Abhängigkeitskette, keine Priorisierung. PR2 zuerst, damit PR1 selbst
schon von einer CI geprüft wird, die auf `dev` hört.

### PR2 (zuerst) — CI-Trigger geradeziehen

**Warum vorne:** `.github/workflows/ci.yml` horcht auf `develop`. Dieser Branch existiert
nicht. Jeder PR *nach* `dev` läuft heute ohne Lint und ohne Tests — auch die vier PRs
dieses Vorhabens.

- `ci.yml`: beide Vorkommen von `develop` → `dev` (in `pull_request.branches` und
  `push.branches`).

**Gate:** PR öffnen → die Checks „Lint with Ruff" und „Run Tests" müssen am PR erscheinen
(heute erscheinen sie nicht).

**Nicht Teil dieses PRs:** das fehlende `--cov` und die dadurch ins Leere laufenden
Codecov-/HTML-Upload-Steps. Das ist der zweite Teil von 2.3 und hat eine eigene
Nebenwirkung: mit `--cov` greift `fail_under = 80` in CI, ein Coverage-Rutsch macht die
Pipeline also rot. Diese Entscheidung gehört nicht in einen Docker-PR.

**Rollback:** ein Revert, keine Abhängigkeiten.

---

### PR1 — `docker-build.yml` + Härtung von `model-service` und `evidently_service`

Das ist der Kern und das Minimum für den 14.08.

#### 1a. Workflow

```yaml
name: Docker - Build & Push to GHCR

on:
  push:
    branches: [dev, main]
    tags: ['v*']
    paths-ignore: ['**.md', '.gitignore']
  pull_request:
    branches: [dev, main]
    paths-ignore: ['**.md', '.gitignore']

concurrency:
  group: docker-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    name: ${{ matrix.name }}
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      fail-fast: false
      matrix:
        include:
          - { name: alligator-model-service,     context: webservice }
          - { name: alligator-evidently-service, context: evidently_service }
          - { name: alligator-prefect,           context: prefect }
```

Entscheidungen im Detail:

- **`fail-fast: false`** — ein defektes Image darf die Ergebnisse der anderen zwei nicht
  abschneiden. Bei einer Härtung, die alle drei Dockerfiles anfasst, ist genau das die
  Information, die man braucht.
- **Image-Referenz kleingeschrieben ausschreiben.** Der Owner heißt `ChrDro`, GHCR-Pfade
  müssen lowercase sein. `${{ github.repository_owner }}` liefert die gemischte
  Schreibweise. `docker/metadata-action` normalisiert seinen `images`-Input selbst, aber
  `docker/login-action` und jede manuelle `docker`-Zeile tun das nicht → im Workflow
  durchgängig `ghcr.io/chrdro/...` literal schreiben.
- **Cache** `type=gha` mit `scope: ${{ matrix.name }}`. Ohne eigenen Scope verdrängen sich
  die drei Images gegenseitig. Das repoweite GHA-Cache-Limit von 10 GB wird bei
  `mode=max` und drei Images erreicht — dann beginnt Verdrängung, kein Fehler. Wenn das
  spürbar wird, ist der nächste Schritt Registry-Cache (`ghcr.io/...:buildcache`), nicht
  weniger Cache.
- **`flavor: latest=false`** in `metadata-action` plus eine explizite `type=raw`-Zeile für
  `latest`. Sonst würde ein `v*`-Tag automatisch auch `latest` setzen und `latest` hätte
  zwei Quellen.

Tag-Matrix:

```yaml
tags: |
  type=ref,event=branch                      # :dev  bzw.  :main
  type=sha,prefix=sha-,format=short          # :sha-aad9c34
  type=semver,pattern={{version}}            # :1.0.0
  type=semver,pattern={{major}}.{{minor}}    # :1.0
  # latest folgt bewusst dev, solange main auf dem Initial Commit steht.
  # Nach dem dev->main-Merge (2.3): 'dev' hier durch 'main' ersetzen - das ist
  # der einzige nötige Eingriff.
  type=raw,value=latest,enable=${{ github.ref == 'refs/heads/dev' }}
```

⚠️ Zu `paths-ignore` am `pull_request`: solange dieser Workflow **kein** Required Check
ist, spart es nur Laufzeit bei Doku-PRs. Würde er später als Required Check eingetragen,
bleiben reine Markdown-PRs ewig auf „Waiting for status" hängen — dann muss `paths-ignore`
raus oder durch einen Skip-Job ersetzt werden.

#### Drei Steps statt einem: bauen → scannen → pushen

Bewusst nicht „build-and-push in einem Schritt":

1. **Build** mit `load: true`, `push: false`, `platforms: linux/amd64`,
   `provenance: false`. `load` legt das Image in den lokalen Docker-Daemon des Runners —
   ohne das hätte Trivy bei einem PR-Lauf gar kein Artefakt zu scannen. `provenance: false`
   verhindert `unknown/unknown`-Attestation-Einträge auf der Paketseite, die in einer Demo
   wie ein Defekt aussehen.
2. **Trivy** auf das lokal geladene Image: `severity: CRITICAL,HIGH` — identisch zum
   Default in [`scripts/scan_vulnerabilities.sh`](../scripts/scan_vulnerabilities.sh),
   damit lokale und CI-Zahlen vergleichbar sind — `exit-code: 0`, zwei Ausgaben (Tabelle
   → `$GITHUB_STEP_SUMMARY`, JSON → Artifact, 30 Tage).
   **Nicht blockierend, mit Absicht:** die Findings des Basis-Images sind nicht durch
   dieses Repository behebbar; ein Gate darauf würde die Pipeline dauerhaft rot halten und
   damit wertlos machen.
3. **Push**, nur wenn `github.event_name != 'pull_request'`: zweiter Aufruf von
   `build-push-action` mit `push: true` und identischen Inputs. Der Build ist dann ein
   vollständiger Cache-Treffer und kostet Sekunden.

Der Nutzen dieser Reihenfolge ist nicht der heutige Lauf, sondern die Option: **kein Image
wird veröffentlicht, bevor es gescannt wurde.** Aus `exit-code: 0` ein `1` zu machen,
sobald die Findings sauber sind, ist damit eine Einzeiler-Änderung und kein Umbau.

#### 1b. Dockerfile-Muster

Angewandt auf `webservice/` und `evidently_service/` (Ports/`CMD`/`libgomp1`
entsprechend). Für `prefect/` gilt es **nicht** — Begründung am Ende dieses Abschnitts.

```dockerfile
FROM python:3.11.13-slim-bookworm AS builder
COPY requirements.txt .
# --prefix statt venv: der Runtime-Stage kopiert /install nach /usr/local, wo
# site-packages und bin ohnehin liegen. Kein PATH-Gebastel, und der Installer
# selbst wandert nicht mit - er lebt nur im Builder.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.13-slim-bookworm AS runtime
# libgomp1 fuer LightGBM, apt-get upgrade fuer die OS-CVEs des Basis-Images (wie
# bisher). Dann fliegt der Installer des Basis-Images raus: ein Serving-Container
# installiert zur Laufzeit nichts. Reihenfolge wheel->setuptools->pip ist Absicht,
# pip entfernt sich zuletzt selbst. Erklaerung siehe Punkt 1 unten.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y wheel setuptools pip \
    && useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=appuser:appuser . /app
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/live')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Vier Punkte, die begründet werden müssen:

1. **Kein Installer im Runtime-Image — und damit drei Pins weniger.** Das Basis-Image
   bringt `pip 24.0`, `setuptools 65.5.1` und `wheel 0.45.1` mit; `wheel 0.45.1` liegt
   *unterhalb* der für CVE-2026-24049 gefixten 0.46.2. Genau deshalb steht in allen drei
   `requirements.txt` ein `wheel==0.47.0` mit dem Kommentar „not used at runtime, pinned
   only to clear CVE-2026-24049". Zwei Mechanismen greifen hier zusammen:
   `python -m pip uninstall` im Runtime-Stage entfernt die drei Pakete des Basis-Images,
   und `--prefix=/install` im Builder kopiert nur die Requirements-Pakete herüber — `wheel`
   landet dort also nur noch, *weil es in der `requirements.txt` steht*. Fällt der Pin,
   ist es weg. **Die CVE-Fläche verschwindet damit, statt gepatcht zu werden**, und
   `setuptools 65.5.1` mit seinen eigenen HIGH-Findings gleich mit.
   ⚠️ Risiko: Pakete, die zur Laufzeit `pkg_resources` importieren, brechen ohne
   `setuptools`. Kandidat ist `gunicorn` in `evidently_service` (ab 23.x auf
   `importlib` umgestellt, also erwartet unkritisch). **Fallback, falls ein Smoke-Test
   bricht:** `setuptools` behalten und stattdessen auf eine gefixte Version pinnen. Die
   `wheel`-Pins fallen in beiden Fällen.
2. **`HEALTHCHECK` auf `/health/live`, nicht `/health/ready`.** Der Image-Default gehört
   auf „Prozess lebt". Der strengere Compose-Healthcheck auf `/health/ready` (verlangt,
   dass die Registry die Modelle wirklich auflöst) bleibt unverändert und überschreibt
   ihn. Zwei Ebenen, zwei Bedeutungen. `evidently_service` hat keine Health-Route → dort
   `GET /tracks`, das beweist, dass Flask antwortet.
3. **`--chown` beim `COPY`, `USER` erst danach**, und `uid 1000` bewusst fest: auf einem
   Linux-Host trifft das die übliche Owner-UID der Bind-Mounts.
4. **Kein `--no-compile`.** Das würde 100–200 MB `.pyc` sparen, aber den in 3.2
   dokumentierten langsamen ersten Request verschlimmern. Für ein Serving-Image der
   falsche Tausch. (`PYTHONDONTWRITEBYTECODE` betrifft nur *zur Laufzeit* neu erzeugte
   `.pyc` in einem ohnehin read-only Codeverzeichnis, nicht die beim Install erzeugten.)

Zusätzlich in `evidently_service/Dockerfile`: `COPY requirements.txt` **vor** dem
`pip install`, `COPY . /app` danach — die letzte offene Cache-Invalidierung aus 3.3.

**`prefect/Dockerfile` bekommt in PR1 nur den `HEALTHCHECK`** (`GET /api/health`), bleibt
sonst single-stage. Grund: das Basis-Image ist `prefecthq/prefect:3.6.25-python3.11`, das
Prefect selbst im System-Python mitbringt. Ein Builder-Stage, der nur die
`requirements.txt`-Extras nach `/install` legt, würde funktionieren — aber nichts
gewinnen, weil die eigentliche Masse (Prefect samt Abhängigkeiten) im Basis-Image steckt
und ohnehin in einer gecachten Layer liegt. Ein zweistufiger Build wäre hier Zeremonie
ohne Wirkung. Der `pip uninstall`-Schritt entfällt aus demselben Grund: bei diesem Image
ist nicht belegt, dass zur Laufzeit niemand `setuptools` braucht — Prefect bringt einen
weit größeren Abhängigkeitsbaum mit als die zwei anderen Services. Der `wheel`-Pin fällt
trotzdem, sobald er nicht mehr durch die Basis-Version unterboten wird (Gate-Schritt 6).
Non-root und `.dockerignore` folgen in PR5.

#### 1c. Compose (nur Namen, kein Verhaltenswechsel)

```yaml
  model-service:
    image: ghcr.io/chrdro/alligator-model-service:dev
    build:
      context: webservice
      dockerfile: Dockerfile
    pull_policy: build     # niemals ziehen; GHCR-Pfad laeuft ueber die Override-Datei
```

Analog für `evidently_service` und `prefect`. `pull_policy: build` friert das heutige
Verhalten explizit ein, statt sich auf Compose-Defaults zu verlassen. Der `image:`-Name
sorgt dafür, dass lokale Builds so heißen wie das Veröffentlichte.

Ebenfalls hier: `evidently_service` bekommt einen Compose-Healthcheck (bisher keiner,
weshalb nichts sinnvoll `depends_on: service_healthy` darauf setzen kann).

**Nebenwirkung, positiv:** `scan_vulnerabilities.sh` liest die Image-Liste aus
`docker compose config --images`. Nach diesem PR stehen dort die GHCR-Namen — der lokale
Trivy-Report benennt also dieselben Artefakte wie die Registry. Voraussetzung bleibt, dass
lokal vorher gebaut wurde (sonst versucht Trivy zu ziehen und braucht ein Login).

#### 1d. Gate für PR1

1. `docker compose build` — alle drei grün
2. `docker images` vor/nach → Zahlen ins PR-Beschreibungsfeld
3. `docker compose up -d` → `model-service`, `prefect`, `evidently_service` erreichen
   `healthy`
4. `docker compose exec model-service id` → **nicht** `uid=0`
5. ein Skript aus [`curl_tests/`](../curl_tests/) liefert eine Prediction — beweist, dass
   der non-root User die MLflow-Artefakte aus MinIO laden kann
6. `bash scripts/scan_vulnerabilities.sh` → Einträge zu `pip`/`setuptools`/`wheel`
   verschwunden. **Erst danach** die `wheel`-Pins aus den drei `requirements.txt` löschen
7. PR öffnen → Actions baut alle drei ohne Push; nach dem Merge nach `dev` erscheinen drei
   Pakete auf der GHCR-Seite

**Größtes Einzelrisiko:** `model-service` hat den Bind-Mount `./mlruns:/app/mlruns`. Als
non-root kann ein Schreibzugriff darauf abgelehnt werden. Unter Docker Desktop für Windows
sind Bind-Mounts UID-unempfindlich, deshalb ist die Erwartung „unproblematisch" — genau
darauf zielen Schritt 3 und 5. **Fallback:** `user: "0:0"` für diesen Service, oder
`mlruns` auf ein Named Volume umstellen.

**Rollback:** Revert des PRs. Bereits gepushte Images bleiben in GHCR liegen und werden
von niemandem gezogen (`pull_policy: build`).

---

### PR3 — Slimming: `mlflow` → `mlflow-skinny`

Der einzige Hebel auf das AP2-Ziel „<500 MB". `mlflow` zieht den kompletten
Server-Stack mit (Flask, gunicorn, alembic, SQLAlchemy, scipy, pyarrow, graphene, docker
…). Der Webservice braucht davon nichts: er spricht über
`mlflow.set_tracking_uri()` mit einem *externen* Tracking-Server und lädt
`models:/<name>@<alias>` als pyfunc.

- `webservice/requirements.txt`: `mlflow==3.14.0` → `mlflow-skinny==3.14.0`.
  `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `boto3` stehen dort bereits
  explizit — die von skinny weggelassenen Laufzeitabhängigkeiten sind also vorhanden.

**Gate — hier hängt der Serving-Pfad dran, also strenger als bei PR1:**

1. `docker images` → tatsächliche Zahl dokumentieren, auch wenn sie über 500 MB liegt
2. Stack hoch, alle fünf `/predict_*`-Routen über `curl_tests/` durch
3. `pytest test/test_models/test_model_schema_contract.py` gegen den laufenden Stack —
   der Test vergleicht MLflow-Signatur gegen die Pydantic-Felder und ist damit genau der
   Nachweis, dass das Laden aus der Registry unverändert funktioniert
4. `pytest` vollständig (die 273 Tests laufen auf dem Host, nicht im Image — sie sichern
   ab, dass die `requirements.txt`-Änderung keine Codeannahme verletzt)

**Fallback:** zurück auf volles `mlflow`, Zahl trotzdem im Plan dokumentieren („<500 MB
mit `mlflow` nicht erreichbar, gemessen: X MB"). Ein belegtes Nein ist hier ein Ergebnis,
kein Fehlschlag.

**Bewusst nicht:** Alpine als Basis. `lightgbm`/`xgboost`/`scipy` haben keine
musl-Wheels → Fallback auf Quellcode-Kompilierung, Buildzeit von Minuten auf Stunden.

---

### PR4 — `evidently_service/references/` mounten statt backen

Heute backt `COPY . /app` die Referenz-CSVs ins Image. Deshalb muss
[`setup_stack.sh`](../scripts/setup_stack.sh) den Service nach jedem Baseline-Bau **neu
bauen** (Schritt 7, um Zeile 388).

- `docker-compose.yaml`: `- ./evidently_service/references:/app/references:ro`
- `evidently_service/.dockerignore`: `references/` ergänzen — und den vorhandenen
  Kommentar, der das Einbacken erklärt, auf den neuen Stand bringen
- `setup_stack.sh`: `docker compose up -d --build evidently_service` →
  `docker compose restart evidently_service`. Ein Restart genügt, weil `init_evidently()`
  beim Import läuft

**Gate:** `setup_stack.sh --rebuild-reference` läuft durch; die Gauge
`evidently_reference_dataset_hash` erscheint danach auf `/metrics` (das Skript prüft das
selbst als Beweis für eine nutzbare Baseline); Rebuild-Schritt ist aus dem Log
verschwunden.

**Nebenwirkung:** Das Image ist ohne Baseline nicht mehr allein lauffähig — die Referenzen
müssen auf dem Host liegen. Das ist der eigentliche Grund, warum dieser Schritt nicht in
PR1 gehört: er verändert, was das veröffentlichte Image *ist*.

---

### PR5 — Prefect non-root, GHCR-Pull-Pfad, Aufräumen, Doku

Der Abschluss-PR. Vier unabhängige Teile; jeder einzeln revertierbar.

**a) `prefect` non-root.** Drei konkrete Hürden: `PREFECT_HOME` (Default `~/.prefect`,
wird beschrieben) auf ein Verzeichnis legen, das `appuser` gehört; die Bind-Mounts
`./prefect` (rw), `./data:ro`, `./src:ro` müssen lesbar bleiben; `setup_stack.sh` ruft
mehrfach `docker compose exec -T prefect python …` auf. Fallback: `user: "0:0"` für den
Service. Außerdem `prefect/.dockerignore` auf das Niveau der anderen zwei bringen
(`.env`, `.env.*`, Cache-Verzeichnisse).

**b) `docker-compose.ghcr.yaml`** — Override, das für die drei Services nur
`pull_policy: always` setzt. Aufruf und die dafür nötige `docker login ghcr.io`-Prozedur
(PAT mit `read:packages`, weil die Pakete privat sind) in die README.

**c) Aufräum-Workflow** — wöchentlich, `actions/delete-package-versions`, behält die
letzten 20 Versionen je Paket. Ohne das wächst die Paketseite pro Push um eine
`sha-`-Version.

**d) Doku** — [MLOPS_PLAN.md](MLOPS_PLAN.md): 2.4 abhaken (inkl. „`GHCR_TOKEN`-Secret:
entfallen"), 3.3 abhaken, AP2-Ist-Stand-Block mit den gemessenen Zahlen ersetzen, den
`.dockerignore`- und `COPY`-Reihenfolge-Befund als bereits vorher erledigt korrigieren.
README: GHCR-Abschnitt. Und ein Satz zu 3.2: *mit* veröffentlichten Images wird
„Contract-Test in CI gegen den echten Stack" von „zu langsam" zu „machbar" — der Punkt
bleibt offen, aber der Grund dagegen ist weg.

**Hier fällt außerdem die Sichtbarkeitsentscheidung** (privat bleiben oder Pakete
öffentlich schalten), weil jetzt die echten Volumina und der Pull-Pfad vorliegen.

---

## 5. Zeitplan

| | bis 14.08. (Präsentation) | danach |
|---|---|---|
| PR2 CI-Trigger | ✅ verbindlich | |
| PR1 Workflow + Härtung | ✅ verbindlich | |
| PR3 Slimming | | ✅ |
| PR4 References-Mount | | ✅ |
| PR5 Prefect/GHCR-Pull/Doku | | ✅ |

Bewusst *nicht* vor dem 14.08.: PR3 fasst den Serving-Pfad an. Zwei Tage vor einer
Präsentation an `mlflow` zu drehen ist das falsche Risiko — der Gewinn wäre nur eine
Zahl auf einer Folie.

Nach PR1+PR2 zeigbar: grüne Actions-Läufe, drei Pakete auf der GHCR-Seite mit
`:dev`/`:latest`/`:sha-…`, die Trivy-Tabelle in der Job-Summary, Vorher/Nachher-Volumina,
und in 2.4 steht `[x]` statt `[ ]`.

---

## 6. Erfolgskriterien gegen AP2

| AP2-Deliverable | Wie es erfüllt wird |
|---|---|
| Docker Image <500 MB | **offen bis PR3 gemessen ist.** Wird nicht versprochen: 1,49 der 1,51 GB sind Python-Abhängigkeiten. Ergebnis wird in jedem Fall mit Zahl dokumentiert |
| Automatischer Push zu GHCR bei main-Branch | erfüllt, mit dokumentierter Abweichung: `latest` folgt `dev`, weil `main` auf dem Initial Commit steht. `main` ist im Trigger enthalten und übernimmt nach dem Merge per Einzeiler |
| Docker Compose nutzt GHCR Images | erfüllt über `docker-compose.ghcr.yaml`, mit dokumentierter Abweichung: **zwei Pfade statt Umschalten**. Lokal bauen bleibt Default, weil ein privates Paket sonst jeden Rechner ohne `docker login` blockiert — inklusive `setup_stack.sh` |

---

## 7. Bewusst nicht im Scope

| Punkt | Warum nicht |
|---|---|
| `dev`→`main`-Merge, Branch Protection | 2.3, Teamentscheidung, fremdes Repository |
| `--cov` in CI + Codecov-Steps reparieren | 2.3; aktiviert zusätzlich das `fail_under = 80`-Gate — eigene Entscheidung, nicht in einem Docker-PR |
| Multi-arch (arm64) | QEMU-Emulation für `scipy`/`lightgbm`/`xgboost`; Plan führt es selbst als „optional" |
| `--no-compile` | verschlimmert die kalte Erstanfrage aus 3.2 |
| Alpine-Basis | keine musl-Wheels für die ML-Abhängigkeiten |
| Multi-stage für `prefect/` | Die Masse steckt im Basis-Image `prefecthq/prefect`, nicht in der `requirements.txt` — zweistufig wäre Zeremonie ohne Wirkung |
| Blockierendes Trivy-Gate | Basis-Image-CVEs sind hier nicht behebbar → dauerhaft rote Pipeline |
| SARIF-Upload ins Security-Tab | braucht GitHub Advanced Security für private Repositories |
| Contract-Test in CI gegen echten Stack (3.2) | wird durch dieses Vorhaben *billiger*, bleibt aber 3.2 |
| `v1.0.0`-Tag setzen | Der `type=semver`-Pfad ist damit nicht praktisch verifiziert. Bewusst offen gelassen — ein Versionsschnitt ist eine Projektentscheidung, keine Build-Entscheidung |

---

## 8. Änderungsübersicht

| Datei | PR | Art |
|---|---|---|
| `.github/workflows/ci.yml` | 2 | `develop` → `dev` |
| `.github/workflows/docker-build.yml` | 1 | neu |
| `webservice/Dockerfile` | 1 | multi-stage, non-root, HEALTHCHECK |
| `evidently_service/Dockerfile` | 1 | dito + `COPY`-Reihenfolge |
| `prefect/Dockerfile` | 1 / 5 | HEALTHCHECK (1), non-root (5) — kein multi-stage, siehe 4/PR1 |
| `docker-compose.yaml` | 1 / 4 | `image:` + `pull_policy` + Healthcheck (1), References-Mount (4) |
| `webservice/requirements.txt` | 1 / 3 | `wheel`-Pin raus (1), `mlflow-skinny` (3) |
| `evidently_service/requirements.txt` | 1 | `wheel`-Pin raus |
| `prefect/requirements.txt` | 1 | `wheel`-Pin raus **nur wenn** der Scan zeigt, dass das Prefect-Basis-Image keine ältere Version mitbringt |
| `evidently_service/.dockerignore` | 4 | `references/` |
| `prefect/.dockerignore` | 5 | `.env`, Cache-Verzeichnisse |
| `scripts/setup_stack.sh` | 4 | Rebuild → Restart |
| `docker-compose.ghcr.yaml` | 5 | neu |
| `.github/workflows/ghcr-cleanup.yml` | 5 | neu |
| `README.md` | 5 | GHCR-Abschnitt |
| `documentation/MLOPS_PLAN.md` | 5 | 2.4, 3.3, AP2 |
