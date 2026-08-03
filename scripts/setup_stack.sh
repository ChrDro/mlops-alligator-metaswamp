#!/usr/bin/env bash
#
# Bring the whole stack from "fresh clone" to "a prediction actually works".
#
# Steps, in order:
#   1. Preflight  - docker, .env, python interpreter
#   2. Compose    - build and start all services, wait until they answer
#   3. Train      - the 5 models, unless they are already registered
#   4. Verify     - registry aliases + the model/schema contract tests
#   5. Smoke test - one real prediction through the API
#   6. Monitoring - build one Evidently baseline per key model (pk/cpk/fk/cfk)
#   7. Streaming  - optional: reset watermarks and fire the push trigger
#
# The script is idempotent: run it again and it skips the training and the
# baseline build it does not need. Use --retrain to force both.
#
#   ./scripts/setup_stack.sh                     # full setup, skips existing models
#   ./scripts/setup_stack.sh --retrain           # retrain all 5 models + rebuild baseline
#   ./scripts/setup_stack.sh --skip-train        # only start + verify + smoke test
#   ./scripts/setup_stack.sh --rebuild-reference # force just the monitoring baseline
#   ./scripts/setup_stack.sh --with-streaming    # also trigger a streaming run (needs Trino)
#
set -euo pipefail

# Always operate from the repo root: the training scripts read data/*.csv relative
# to the current working directory, so running this from anywhere else fails oddly.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RETRAIN=false
SKIP_TRAIN=false
WITH_STREAMING=false
REBUILD_REFERENCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --retrain)            RETRAIN=true ;;
        --skip-train)         SKIP_TRAIN=true ;;
        --with-streaming)     WITH_STREAMING=true ;;
        --rebuild-reference)  REBUILD_REFERENCE=true ;;
        # Print the header comment block, stopping at the first line of real code,
        # so --help cannot drift out of sync with a hard-coded line range.
        -h|--help)
            awk 'NR>1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# --- output helpers ----------------------------------------------------------

if [ -t 1 ]; then
    BOLD=$(printf '\033[1m'); GREEN=$(printf '\033[32m'); RED=$(printf '\033[31m')
    YELLOW=$(printf '\033[33m'); RESET=$(printf '\033[0m')
else
    BOLD=""; GREEN=""; RED=""; YELLOW=""; RESET=""
fi

step()  { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$RESET"; }
ok()    { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn()  { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
fail()  { printf '\n%sError:%s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# --- 1. preflight ------------------------------------------------------------

step "1/7  Preflight"

command -v docker >/dev/null 2>&1 || fail "docker not found."
docker info >/dev/null 2>&1 || fail "Docker daemon not running. Start Docker Desktop."
ok "Docker found and running"

[ -f .env ] || fail ".env missong. Please create and fill .env from .env.template."

# shellcheck disable=SC1091  # .env is user config, not tracked
set -a; . ./.env; set +a

for var in TRINO_USERNAME TRINO_PASSWORD TRINO_IP_ADDRESS MINIO_ROOT_USER MINIO_ROOT_PASSWORD POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB; do
    [ -n "${!var:-}" ] || fail ".env: $var not set."
done
ok ".env found and complete"

# A venv created on Windows puts the interpreter in Scripts/, not bin/, so check
# both before falling back to uv.
if [ -x .venv/bin/python ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
    warn "No .venv found, use 'uv run python' before executing python"
else
    fail "Neother .venv/bin/python nor uv found. Execute 'uv sync' to create .venv."
fi
ok "Python: $PYTHON"

# The training scripts and the contract tests run on the HOST, where the compose
# hostnames (mlflow:5000, minio:9000) do not resolve. .env carries those names for
# the containers, so everything host-side has to be pointed at localhost instead.
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="$MINIO_ROOT_USER"
export AWS_SECRET_ACCESS_KEY="$MINIO_ROOT_PASSWORD"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-central-1}"
ok "Switched MLflow/S3 to localhost (Container still use names in .env)"

# The training output below is redirected to a log file, so on Windows Python
# picks the ANSI codepage (cp1252) for stdout instead of the console's UTF-8.
# MLflow prints "<runner emoji> View run ... at: ..." after every run, which then
# dies with UnicodeEncodeError. UTF-8 mode makes the redirect encoding-safe.
export PYTHONUTF8=1

# --- helpers -----------------------------------------------------------------

wait_for_http() {  # url, label, timeout_seconds
    local url=$1 label=$2 timeout=$3 waited=0
    printf '    waiting for %s ' "$label"
    until curl -sf -o /dev/null "$url" 2>/dev/null; do
        if [ "$waited" -ge "$timeout" ]; then
            printf '\n'
            fail "$label after ${timeout} seconds not accessible ($url). Logs: docker compose logs ${label%% *}"
        fi
        printf '.'
        sleep 3
        waited=$((waited + 3))
    done
    printf '\n'
    ok "$label accessible (${waited}s)"
}

wait_for_healthy() {  # service, timeout_seconds
    local svc=$1 timeout=$2 waited=0 cid status
    printf '    waiting for %s ' "$svc"
    while true; do
        cid="$(docker compose ps -q "$svc" 2>/dev/null || true)"
        if [ -n "$cid" ]; then
            status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}nohealth{{end}}' "$cid" 2>/dev/null || echo starting)"
            [ "$status" = "healthy" ] && { printf '\n'; ok "$svc healthy (${waited}s)"; return 0; }
            [ "$status" = "nohealth" ] && { printf '\n'; ok "$svc running (no healthcheck)"; return 0; }
        fi
        if [ "$waited" -ge "$timeout" ]; then
            printf '\n'
            fail "$svc after ${timeout} seconds not healthy. Logs: docker compose logs $svc"
        fi
        printf '.'
        sleep 3
        waited=$((waited + 3))
    done
}

registered_models() {
    $PYTHON - <<'PY' 2>/dev/null || true
import mlflow
from mlflow.tracking import MlflowClient

try:
    for model in MlflowClient().search_registered_models():
        if "dev" in (model.aliases or {}):
            print(model.name)
except Exception:
    pass
PY
}

# --- 2. compose --------------------------------------------------------------

step "2/7  Build and start container"

# --build matters: webservice/ is baked into the image, not mounted. Without it a
# code change silently keeps serving the old routes.
docker compose up -d --build
ok "docker compose up executed"

wait_for_healthy postgres 120
wait_for_healthy minio 120
wait_for_http "http://localhost:5000/health" "mlflow" 180
wait_for_http "http://localhost:8080/" "model-service" 180

# The prefect container pip-installs its dependencies before the server starts, so
# the first boot after a rebuild is slow. Its healthcheck allows 300s start period.
wait_for_http "http://localhost:4200/api/health" "prefect" 420

# The task_4 training script asks this service to name each discovered subject area,
# so it has to be up before step 3. Its healthcheck only passes once the model is
# pulled into the volume, which on a cold start means a multi-GB download.
if [ "$SKIP_TRAIN" = false ]; then
    wait_for_healthy ollama 900
fi

# --- 3. training -------------------------------------------------------------

TRAIN_SCRIPTS=(
    "pk_model:task_1/task_1_pk_train_and_register.py"
    "composite_pk_model:task_1/task_1_cpk_train_and_register.py"
    "fk_model:task_2/task_2_fk_train_and_register.py"
    "composite_fk_model:task_2/task_2_cfk_train_and_register.py"
    "denormalization_model:task_3/task_3_denormalization_train_and_register.py"
    # Slowest of the five on a cold cache: it downloads the sentence-transformer,
    # embeds every table, and asks the ollama service to name each cluster. Step 5
    # globs curl_tests/test_curl_predict_*.sh, so leaving this untrained would fail
    # the smoke test rather than skip it.
    "subject_area_model:task_4/task_4_subject_area_train_and_register.py"
)

step "3/7  Train and register models"

if [ "$SKIP_TRAIN" = true ]; then
    warn "--skip-train was set, skipped training"
else
    EXISTING="$(registered_models)"
    for entry in "${TRAIN_SCRIPTS[@]}"; do
        model_name="${entry%%:*}"
        script="${entry#*:}"

        if [ "$RETRAIN" = false ] && printf '%s\n' "$EXISTING" | grep -qx "$model_name"; then
            ok "$model_name already registered, skipped (--retrain erzwingt neu)"
            continue
        fi

        printf '    training %s ... ' "$model_name"
        log="$(mktemp)"
        if $PYTHON "$script" >"$log" 2>&1; then
            printf '%s✓%s\n' "$GREEN" "$RESET"
            rm -f "$log"
        else
            printf '%s✗%s\n' "$RED" "$RESET"
            tail -25 "$log" >&2
            fail "Training of $model_name failed (full log: $log)"
        fi
    done
fi

# --- 4. verify ---------------------------------------------------------------

step "4/7  Check Registry und Pydantic schema contract prüfen"

MISSING=""
FOUND="$(registered_models)"
for entry in "${TRAIN_SCRIPTS[@]}"; do
    model_name="${entry%%:*}"
    if printf '%s\n' "$FOUND" | grep -qx "$model_name"; then
        ok "$model_name @dev"
    else
        MISSING="$MISSING $model_name"
    fi
done
[ -z "$MISSING" ] || fail "Not registered:$MISSING — without models each predict returns code 400."

# Catches the drift that once let three of four models return NULL in production:
# the Pydantic request schema and the logged model signature must agree on the
# feature set AND their order.
if $PYTHON -m pytest test/test_models -q >/dev/null 2>&1; then
    ok "Passed pydantic schema contract test"
else
    $PYTHON -m pytest test/test_models -q 2>&1 | tail -20 >&2
    fail "Schema drift between Pydantic models und model registry. More details above."
fi

# --- 5. smoke test -----------------------------------------------------------

step "5/7  Smoke test: Real Predict with API"

for script in curl_tests/test_curl_predict_*.sh; do
    endpoint="$(basename "$script" .sh | sed 's/^test_curl_//')"
    response="$(bash "$script" 2>/dev/null || true)"
    if printf '%s' "$response" | grep -q '"prediction"'; then
        value="$(printf '%s' "$response" | sed -n 's/.*"prediction":\([^,}]*\).*/\1/p')"
        ok "$endpoint -> prediction=$value"
    else
        printf '%s\n' "$response" >&2
        fail "$endpoint returns no predicts. See logs at: docker compose logs model-service"
    fi
done

# --- 6. monitoring baseline --------------------------------------------------
#
# Ordering here is not free to change. The baseline is a join of two things that
# only both exist at this point in the script:
#   - ground truth, from data/summary_output_task_1_2_training.csv
#   - predictions, from a registered pk_model served over HTTP (steps 3 and 5)
# And because evidently_service bakes its files in with `COPY . /app` rather than
# mounting them, writing the CSV is not enough - the image has to be rebuilt after.
# That is why this cannot move up next to the other builds in step 2.

step "6/7  Evidently monitoring baselines"

REFERENCE_DIR="evidently_service/references"
HOLDOUT_DIR="data/holdouts"
# One reference + one holdout per monitored model.
MONITORED_TRACKS="pk_columns cpk_columns fk_columns cfk_columns"

# A retrain invalidates the baseline: it describes how one specific model version
# scored, so comparing a new model against it would measure the version change
# rather than anything about the data.
if [ "$RETRAIN" = true ]; then
    REBUILD_REFERENCE=true
    warn "Models were retrained - baseline must be rebuilt to match"
fi

for track in $MONITORED_TRACKS; do
    if [ ! -f "$REFERENCE_DIR/$track.csv" ] || [ ! -f "$HOLDOUT_DIR/$track.csv" ]; then
        REBUILD_REFERENCE=true
        break
    fi
done

if [ "$REBUILD_REFERENCE" = false ]; then
    ok "Baselines exist for all 4 tracks, skipped (--rebuild-reference forces it)"
else
    printf '    scoring labelled rows through all 4 predict endpoints (~2min) ... '
    log="$(mktemp)"
    if $PYTHON evidently_service/build_monitoring_references.py \
        --model-url "http://localhost:8080" >"$log" 2>&1; then
        printf '%s✓%s\n' "$GREEN" "$RESET"
        # Surface the per-track accuracy lines: a baseline built against a broken
        # model would otherwise look like a success and quietly poison every
        # comparison downstream.
        sed -n 's/.*\(\(pk\|cpk\|fk\|cfk\)_columns  *ref=.*\)/      \1/p' "$log" || true
        rm -f "$log"
    else
        printf '%s✗%s\n' "$RED" "$RESET"
        tail -20 "$log" >&2
        fail "Could not build the monitoring baseline (full log: $log)"
    fi

    # COPY . /app means the CSVs only reach the container through a rebuild.
    printf '    rebuilding evidently_service to bake in the baseline ... '
    if docker compose up -d --build evidently_service >/dev/null 2>&1; then
        printf '%s✓%s\n' "$GREEN" "$RESET"
    else
        printf '%s✗%s\n' "$RED" "$RESET"
        fail "evidently_service rebuild failed. Logs: docker compose logs evidently_service"
    fi
fi

wait_for_http "http://localhost:8085/metrics" "evidently_service" 120

# The reference gauges are exported at startup, so their absence means the service
# came up without a usable baseline - the dashboards would then show a current
# series with nothing to compare it against.
ACTIVE_TRACKS="$(curl -s "http://localhost:8085/metrics" \
    | sed -n 's/^evidently_reference_dataset_hash{dataset_name="\([^"]*\)".*/\1/p' | sort -u | tr '\n' ' ')"
if [ -n "$ACTIVE_TRACKS" ]; then
    ok "Active monitoring tracks: $ACTIVE_TRACKS"
else
    warn "No active tracks - check: docker compose logs evidently_service"
fi

# --- 7. streaming (optional) -------------------------------------------------

step "7/7  Streaming-Trigger"

if [ "$WITH_STREAMING" = false ]; then
    warn "skipped — set argument --with-streaming (needs accessible Trino instance)"
else
    [ -n "${TRINO_IP_ADDRESS:-}" ] || fail "TRINO_IP_ADDRESS missing in .env, Trino needed for streaming."

    # Dropping the watermark table makes the detector treat every table as new, so
    # the run below actually has work to do instead of finding nothing changed.
    docker compose exec -T prefect python -c "
from change_detector import get_trino_engine
from sqlalchemy import text
with get_trino_engine().begin() as conn:
    conn.execute(text('DROP TABLE IF EXISTS duckdb.staging.source_watermarks'))
print('Watermarks reseted')
" || fail "Trino unreachable. Check network."
    ok "Watermarks reseted — next run will assume all talbes als "new""

    bash trigger_prefect_pipeline.sh >/dev/null 2>&1 \
        || fail "Webhook call failed."
    ok "Push trigger send"
    warn "Runs need may need one or two minutes — see progress at: http://localhost:4200/runs"
fi

# --- summary -----------------------------------------------------------------

cat <<EOF

$BOLD$GREEN Stack is ready.$RESET

  MLflow        http://localhost:5000
  Model API     http://localhost:8080/docs
  Prefect       http://localhost:4200
  Grafana       http://localhost:3000   (GF_SECURITY_ADMIN_USER / GF_SECURITY_ADMIN_PASSWORD from .env)
  Prometheus    http://localhost:9090
  Alertmanager  http://localhost:9093   (Null Receiver, sends nothing)
  MinIO         http://localhost:9001
  Evidently     http://localhost:8085/tracks          (what is monitored)
                http://localhost:8085/report/<track> (drift HTML, e.g. pk_columns)

  Dashboards:
    Model Service - Golden Signals            /d/model-service-golden-signals
    Key Predict Data Drift Monitoring         /d/evidently-key-drift
    Key Model Performance Monitoring          /d/evidently-model-quality
    Normalform Predict Data Drift Monitoring  /d/evidently-normalform-drift
    Normalform Model Performance Monitoring   /d/evidently-normalform-quality

  Ollama        http://localhost:${OLLAMA_HOST_PORT:-11434}   (names the task_4 subject areas)

  Single Predict:       bash curl_tests/test_curl_predict_pk.sh
  Subject area:         bash curl_tests/test_curl_predict_subject_area.sh
  Trigger Streaming:    bash trigger_prefect_pipeline.sh
  Results:              duckdb.prediction_results.key_results / nf_results

  Monitoring notes:
    Drift fills automatically from all 5 predict endpoints, one track per
    model. Each needs service.window_size predictions before its first
    report appears. The key dashboards carry a Model dropdown (pk/cpk/fk/cfk);
    normalform has its own pair of dashboards because it is multiclass.
    F1/precision/recall need ground-truth labels, which live predictions do
    not carry, so they come from the hourly model-quality-backtest
    deployment. Run it now instead of waiting for :17 with:
      docker compose exec -T -w /opt/flows prefect python model_quality_backtest.py
    What is actually being monitored:  curl localhost:8085/tracks

EOF
