#!/bin/bash
# Run the V1 pre-refactor benchmark suite and save a JSON baseline into
# old_tests/baselines/.
#
# Usage:
#   bash old_tests/run_baseline.sh             # label = <yyyy-mm-dd>_<git sha>
#   bash old_tests/run_baseline.sh my_label    # explicit label

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Pick label.
if [[ $# -ge 1 && -n "$1" ]]; then
  LABEL="$1"
else
  DATE="$(date +%Y-%m-%d)"
  if git rev-parse --short=8 HEAD >/dev/null 2>&1; then
    SHA="$(git rev-parse --short=8 HEAD)"
    LABEL="${DATE}_${SHA}"
  else
    LABEL="${DATE}_nogit"
  fi
fi

# Pick Python, same logic as install.sh.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "No Python interpreter found. Set PYTHON=... or create a .venv." >&2
    exit 1
  fi
fi

BASELINE_DIR="old_tests/baselines"
mkdir -p "${BASELINE_DIR}"
OUT="${BASELINE_DIR}/${LABEL}.json"

echo "Python: ${PYTHON}"
echo "Baseline: ${OUT}"
echo

"${PYTHON}" -m pytest old_tests/bench_generator.py \
  --benchmark-only \
  --benchmark-columns=mean,median,stddev,min,max,ops,rounds \
  --benchmark-json="${OUT}" \
  -q

echo
echo "Wrote ${OUT}"
