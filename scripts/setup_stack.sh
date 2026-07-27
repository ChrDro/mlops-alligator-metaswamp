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
#   6. Streaming  - optional: reset watermarks and fire the push trigger
#
# The script is idempotent: run it again and it skips the training it does not
# need. Use --retrain to force it.
#
#   ./scripts/setup_stack.sh                  # full setup, skips existing models
#   ./scripts/setup_stack.sh --retrain        # retrain all 5 models
#   ./scripts/setup_stack.sh --skip-train     # only start + verify + smoke test
#   ./scripts/setup_stack.sh --with-streaming # also trigger a streaming run (needs Trino)
#
set -euo pipefail

# Always operate from the repo root: the training scripts read data/*.csv relative
# to the current working directory, so running this from anywhere else fails oddly.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RETRAIN=false
SKIP_TRAIN=false
WITH_STREAMING=false

while [ $# -gt 0 ]; do
    case "$1" in
        --retrain)        RETRAIN=true ;;
        --skip-train)     SKIP_TRAIN=true ;;
        --with-streaming) WITH_STREAMING=true ;;
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
fail()  { printf '\n%sFEHLER:%s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# --- 1. preflight ------------------------------------------------------------

step "1/6  Preflight"

command -v docker >/dev/null 2>&1 || fail "docker nicht gefunden."
docker info >/dev/null 2>&1 || fail "Docker-Daemon läuft nicht. Docker Desktop starten."
ok "Docker läuft"

[ -f .env ] || fail ".env fehlt. Aus .env.template erzeugen und ausfüllen."

# shellcheck disable=SC1091  # .env is user config, not tracked
set -a; . ./.env; set +a

for var in MINIO_ROOT_USER MINIO_ROOT_PASSWORD POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB; do
    [ -n "${!var:-}" ] || fail ".env: $var ist nicht gesetzt."
done
ok ".env vollständig"

if [ -x .venv/bin/python ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON="uv run python"
    warn "Kein .venv gefunden, nutze 'uv run python'"
else
    fail "Weder .venv/bin/python noch uv gefunden. Erst 'uv sync' ausführen."
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
ok "MLflow/S3 auf localhost umgebogen (Container nutzen weiter die .env-Namen)"

# --- helpers -----------------------------------------------------------------

wait_for_http() {  # url, label, timeout_seconds
    local url=$1 label=$2 timeout=$3 waited=0
    printf '    warte auf %s ' "$label"
    until curl -sf -o /dev/null "$url" 2>/dev/null; do
        if [ "$waited" -ge "$timeout" ]; then
            printf '\n'
            fail "$label nach ${timeout}s nicht erreichbar ($url). Logs: docker compose logs ${label%% *}"
        fi
        printf '.'
        sleep 3
        waited=$((waited + 3))
    done
    printf '\n'
    ok "$label erreichbar (${waited}s)"
}

wait_for_healthy() {  # service, timeout_seconds
    local svc=$1 timeout=$2 waited=0 cid status
    printf '    warte auf %s ' "$svc"
    while true; do
        cid="$(docker compose ps -q "$svc" 2>/dev/null || true)"
        if [ -n "$cid" ]; then
            status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}nohealth{{end}}' "$cid" 2>/dev/null || echo starting)"
            [ "$status" = "healthy" ] && { printf '\n'; ok "$svc healthy (${waited}s)"; return 0; }
            [ "$status" = "nohealth" ] && { printf '\n'; ok "$svc läuft (kein Healthcheck)"; return 0; }
        fi
        if [ "$waited" -ge "$timeout" ]; then
            printf '\n'
            fail "$svc nach ${timeout}s nicht healthy. Logs: docker compose logs $svc"
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

step "2/6  Container bauen und starten"

# --build matters: webservice/ is baked into the image, not mounted. Without it a
# code change silently keeps serving the old routes.
docker compose up -d --build
ok "docker compose up abgesetzt"

wait_for_healthy postgres 120
wait_for_healthy minio 120
wait_for_http "http://localhost:5000/health" "mlflow" 180
wait_for_http "http://localhost:8080/" "model-service" 180

# The prefect container pip-installs its dependencies before the server starts, so
# the first boot after a rebuild is slow. Its healthcheck allows 300s start period.
wait_for_http "http://localhost:4200/api/health" "prefect" 420

# --- 3. training -------------------------------------------------------------

TRAIN_SCRIPTS=(
    "pk_model:task_1/task_1_pk_train_and_register.py"
    "composite_pk_model:task_1/task_1_cpk_train_and_register.py"
    "fk_model:task_2/task_2_fk_train_and_register.py"
    "composite_fk_model:task_2/task_2_cfk_train_and_register.py"
    "denormalization_model:task_3/task_3_denormalization_train_and_register.py"
)

step "3/6  Modelle trainieren und registrieren"

if [ "$SKIP_TRAIN" = true ]; then
    warn "--skip-train gesetzt, Training übersprungen"
else
    EXISTING="$(registered_models)"
    for entry in "${TRAIN_SCRIPTS[@]}"; do
        model_name="${entry%%:*}"
        script="${entry#*:}"

        if [ "$RETRAIN" = false ] && printf '%s\n' "$EXISTING" | grep -qx "$model_name"; then
            ok "$model_name bereits registriert, übersprungen (--retrain erzwingt neu)"
            continue
        fi

        printf '    trainiere %s ... ' "$model_name"
        log="$(mktemp)"
        if $PYTHON "$script" >"$log" 2>&1; then
            printf '%s✓%s\n' "$GREEN" "$RESET"
            rm -f "$log"
        else
            printf '%s✗%s\n' "$RED" "$RESET"
            tail -25 "$log" >&2
            fail "Training von $model_name fehlgeschlagen (vollständiges Log: $log)"
        fi
    done
fi

# --- 4. verify ---------------------------------------------------------------

step "4/6  Registry und Schema-Verträge prüfen"

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
[ -z "$MISSING" ] || fail "Nicht registriert:$MISSING — ohne diese Modelle liefert jeder Predict 400."

# Catches the drift that once let three of four models return NULL in production:
# the Pydantic request schema and the logged model signature must agree on the
# feature set AND their order.
if $PYTHON -m pytest test/test_models -q >/dev/null 2>&1; then
    ok "Schema-Contract-Tests bestanden"
else
    $PYTHON -m pytest test/test_models -q 2>&1 | tail -20 >&2
    fail "Schema-Drift zwischen Pydantic-Modellen und Registry. Details oben."
fi

# --- 5. smoke test -----------------------------------------------------------

step "5/6  Smoke-Test: Real Predict with API"

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

# --- 6. streaming (optional) -------------------------------------------------

step "6/6  Streaming-Trigger"

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

    bash curl_tests/test_curl_notify_new_data.sh >/dev/null 2>&1 \
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
  Grafana       http://localhost:3000   (currently without Dashboards)
  Prometheus    http://localhost:9090
  Alertmanager  http://localhost:9093   (Null-Receiver, sends nothing)
  MinIO         http://localhost:9001

  Single Predict:       bash curl_tests/test_curl_predict_pk.sh
  Trigger Streaming:    bash curl_tests/test_curl_notify_new_data.sh
  Results:              duckdb.prediction_results.key_results / nf_results

EOF
