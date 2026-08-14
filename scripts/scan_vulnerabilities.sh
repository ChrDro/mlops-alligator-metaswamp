#!/usr/bin/env bash
# Scans every image used by the docker-compose stack with Trivy and renders
# a consolidated PDF report. Requires: trivy, docker compose, python3+reportlab.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEVERITY="${SEVERITY:-CRITICAL,HIGH}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="security-reports/${TIMESTAMP}"
JSON_DIR="${OUT_DIR}/raw"
mkdir -p "${JSON_DIR}"

echo "Collecting images from docker-compose.yaml ..."
IMAGES=()
while IFS= read -r line; do
  [ -n "${line}" ] && IMAGES+=("${line}")
done < <(docker compose config --images | sort -u)

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "No images found via 'docker compose config --images'." >&2
  exit 1
fi

echo "Found ${#IMAGES[@]} unique image(s):"
printf '  - %s\n' "${IMAGES[@]}"

echo "Refreshing Trivy's vulnerability database ..."
trivy image --download-db-only

FAILED=()
for image in "${IMAGES[@]}"; do
  safe_name="$(echo "${image}" | tr '/:' '__')"
  echo "Scanning ${image} (severity: ${SEVERITY}) ..."
  if ! trivy image \
        --severity "${SEVERITY}" \
        --format json \
        --output "${JSON_DIR}/${safe_name}.json" \
        --quiet \
        "${image}"; then
    echo "  -> scan failed for ${image}, continuing" >&2
    FAILED+=("${image}")
  fi
done

echo "Rendering PDF report ..."
PYTHON_BIN="$(command -v python3)"
if ! "${PYTHON_BIN}" -c "import reportlab" >/dev/null 2>&1; then
  echo "reportlab not found in ${PYTHON_BIN}, installing ..."
  "${PYTHON_BIN}" -m pip install --quiet reportlab
fi
"${PYTHON_BIN}" scripts/generate_vulnerability_report.py "${JSON_DIR}" "${OUT_DIR}/vulnerability-report.pdf"

echo
echo "Done. Report: ${OUT_DIR}/vulnerability-report.pdf"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Warning: ${#FAILED[@]} image(s) failed to scan: ${FAILED[*]}" >&2
fi
