#!/usr/bin/env bash
#
# Run docker-compose.yaml from inside a container (Docker-out-of-Docker).
#
# The launcher container defined in docker-compose.launcher.yaml gets the host's
# Docker socket mounted, so the compose CLI inside it drives the host daemon.
# The stack it brings up is identical to `docker compose up` on the host: same
# published ports, same named volumes, same image cache. There is no second
# Docker daemon and no nesting.
#
# Every argument is appended to `docker compose` inside the launcher:
#
#   ./scripts/run_stack_in_container.sh                  # up -d --build (default)
#   ./scripts/run_stack_in_container.sh ps               # status of the stack
#   ./scripts/run_stack_in_container.sh logs -f grafana  # follow one service
#   ./scripts/run_stack_in_container.sh up -d prometheus grafana
#   ./scripts/run_stack_in_container.sh down             # stop the stack
#   ./scripts/run_stack_in_container.sh --shell          # shell in the launcher
#
# Runs on macOS, Linux and Windows (Git Bash / WSL2) with Docker Desktop.
# Requires a .env file: compose interpolates MINIO_*, POSTGRES_* and GF_* from it.
#
set -euo pipefail

# The launcher mounts the repo by absolute path, so resolve that from here
# rather than trusting the caller's working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LAUNCHER_FILE="docker-compose.launcher.yaml"
# Deliberately not the main stack's project name: the launcher would otherwise
# join that project, show up in its `ps` output, and count as an orphan
# container to the compose run happening inside it.
LAUNCHER_PROJECT="mlops-alligator-metaswamp-launcher"
LAUNCHER_IMAGE="docker:28-cli"

OPEN_SHELL=false

if [ $# -gt 0 ]; then
    case "$1" in
        # Print the header comment block, stopping at the first line of real
        # code, so --help cannot drift out of sync with a hard-coded range.
        -h|--help)
            awk 'NR>1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
            exit 0 ;;
        --shell)
            OPEN_SHELL=true
            shift ;;
    esac
fi

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

IS_WINDOWS=false
case "${OSTYPE:-}" in
    msys*|cygwin*|win32*) IS_WINDOWS=true ;;
esac

# --- preflight ---------------------------------------------------------------

step "Preflight"

command -v docker >/dev/null 2>&1 || fail "docker not found."
docker info >/dev/null 2>&1 || fail "Docker daemon not running. Start Docker Desktop."
ok "Docker daemon reachable"

if [ "$IS_WINDOWS" = true ]; then
    # Git Bash rewrites anything that looks like a unix path in an argument, so
    # -v /var/run/docker.sock:... would arrive as C:/Program Files/Git/var/...
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL='*'

    # Not the Windows path: a drive letter carries a colon, and compose splits
    # the short volume syntax on colons (the same trap as the ${PWD} comment in
    # docker-compose.yaml). /mnt/<drive>/... is what the daemon in Docker
    # Desktop's WSL2 backend uses for the host filesystem, and it has no colon.
    if [ -z "${PROJECT_DIR:-}" ]; then
        win_root="$(pwd -W 2>/dev/null | tr '\\' '/' || true)"
        case "$win_root" in
            ?:/*)
                drive="$(printf '%s' "$win_root" | cut -c1 | tr 'A-Z' 'a-z')"
                PROJECT_DIR="/mnt/${drive}${win_root#?:}" ;;
            *)
                # Already a unix path: a WSL2 shell rather than Git Bash.
                PROJECT_DIR="$REPO_ROOT" ;;
        esac
    fi

    # A host path would be meaningless here - on Windows the socket lives inside
    # Docker Desktop's Linux VM, which is also where the launcher runs.
    DOCKER_SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"
    ok "Windows detected, using $DOCKER_SOCKET"
else
    # Overridable for the rare setup whose daemon sees the repo under a
    # different path, e.g. a remote or rootless daemon.
    PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"

    # Ask the CLI which socket it talks to rather than assuming
    # /var/run/docker.sock: Docker Desktop, Colima and Rancher Desktop each put
    # it somewhere else, and only the active context knows which one is live.
    if [ -z "${DOCKER_SOCKET:-}" ]; then
        endpoint="$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)"
        case "$endpoint" in
            unix://*) DOCKER_SOCKET="${endpoint#unix://}" ;;
            "")       DOCKER_SOCKET="/var/run/docker.sock" ;;
            *)        fail "The active Docker context talks to '$endpoint', not a unix socket.
       Docker-out-of-Docker mounts a socket into the launcher, so point
       DOCKER_SOCKET at one explicitly if the daemon exposes it." ;;
        esac
    fi
    [ -S "$DOCKER_SOCKET" ] || fail "No Docker socket at $DOCKER_SOCKET (override with DOCKER_SOCKET=...)."
    ok "Socket $DOCKER_SOCKET"
fi

[ -f .env ] || fail "No .env in $REPO_ROOT. Copy it from .env.example and fill in the values."
ok ".env present"

# The one assumption worth testing instead of documenting: that PROJECT_DIR is a
# path the *daemon* resolves. docker-compose.yaml's bind mounts (./prometheus,
# ./grafana, ./mlruns, ./data) are resolved by the CLI in the launcher but
# mounted by the host daemon, so a wrong dialect here would not fail loudly - it
# would start Prometheus and Grafana on empty directories.
if ! docker run --rm -v "${PROJECT_DIR}:${PROJECT_DIR}" "$LAUNCHER_IMAGE" \
        sh -c "test -f '${PROJECT_DIR}/docker-compose.yaml'" >/dev/null 2>&1; then
    fail "The daemon cannot see this repo at $PROJECT_DIR.
       Bind mounts inside the launcher would resolve to empty directories, so
       Prometheus and Grafana would come up without their configuration.
       Set PROJECT_DIR to the path your Docker daemon uses for $REPO_ROOT, or
       start the stack directly with 'docker compose up --build'."
fi
ok "Daemon resolves the repo at $PROJECT_DIR"

# --- run ---------------------------------------------------------------------

# Consumed by docker-compose.launcher.yaml.
export PROJECT_DIR
export DOCKER_SOCKET

# `docker compose run` allocates a TTY when it has one and aborts when it does
# not, which would break this script in CI and in git hooks. Git Bash reports a
# TTY that Docker cannot use ("the input device is not a TTY"), so never ask for
# one there. Written as two calls rather than a flag array on purpose: macOS
# ships bash 3.2, where expanding an empty array under `set -u` is an error.
USE_TTY=false
if [ -t 0 ] && [ "$IS_WINDOWS" = false ]; then
    USE_TTY=true
fi

launch() {
    if [ "$USE_TTY" = true ]; then
        docker compose -p "$LAUNCHER_PROJECT" -f "$LAUNCHER_FILE" run --rm stack-launcher "$@"
    else
        docker compose -p "$LAUNCHER_PROJECT" -f "$LAUNCHER_FILE" run --rm -T stack-launcher "$@"
    fi
}

if [ "$OPEN_SHELL" = true ]; then
    step "Opening a shell in the launcher (the stack is at $PROJECT_DIR)"
    [ "$IS_WINDOWS" = true ] && warn "Git Bash needs a prompt: winpty ./scripts/run_stack_in_container.sh --shell"
    launch sh
    exit $?
fi

if [ $# -eq 0 ]; then
    step "Starting the stack from inside the launcher container"
    # No arguments: docker-compose.launcher.yaml's own command runs (up -d --build).
    launch
    ok "Stack started"

    printf '\n    %-38s %s\n' "Prediction API (Swagger docs)" "http://localhost:8080/docs"
    printf '    %-38s %s\n'   "Prometheus"                    "http://localhost:9090"
    printf '    %-38s %s\n'   "Grafana"                       "http://localhost:3000"
    printf '    %-38s %s\n'   "Evidently drift report"        "http://localhost:8085/report"
    printf '    %-38s %s\n'   "MLflow"                        "http://localhost:5000"
    printf '    %-38s %s\n'   "Prefect"                       "http://localhost:4200"
    warn "Prefect installs its dependencies on first boot and needs a few minutes."
    printf '\n    Next: ./scripts/setup_stack.sh --skip-train  (verify + smoke test)\n'
    printf '    Stop: ./scripts/run_stack_in_container.sh down\n\n'
else
    step "docker compose $* (inside the launcher container)"
    launch docker compose "$@"
fi
