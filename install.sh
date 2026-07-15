#!/bin/bash
set -euo pipefail

# GraphGen build shim.
#
# Post-Step-1a: CMake + scikit-build-core do all the real work. This script
# just picks a Python interpreter and runs an editable install, which:
#   1) configures + builds the C++ extension via CMake,
#   2) drops a rebuild-on-import shim into site-packages
#      (see tool.scikit-build.editable.rebuild in pyproject.toml),
# so `import generator` from anywhere transparently rebuilds when C++ sources
# change.
#
# Callers can override the interpreter with `PYTHON=/path/to/python bash install.sh`.

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
echo "Using Python: ${PYTHON}"

# uv-managed venvs don't ship pip, so prefer `uv pip` when available.
# Falls back to `python -m pip` otherwise. Both invoke scikit-build-core
# through the pyproject.toml [build-system] block.
#
# --no-build-isolation is important: without it, uv builds in an isolated
# throwaway env and CMake bakes that env's ninja path into CMakeCache.txt,
# breaking the editable rebuild shim later at import time. Using the
# persistent venv (which pins scikit-build-core, cmake, ninja, pybind11 as
# runtime deps) means the rebuild shim finds the same tools every time.
if command -v uv >/dev/null 2>&1; then
  # Make sure build deps exist in the venv before we skip isolation.
  VIRTUAL_ENV="$(cd "$(dirname "${PYTHON}")/.." && pwd)" \
    uv pip install --python "${PYTHON}" \
      "scikit-build-core>=0.10" "cmake>=3.24" ninja "pybind11>=2.13"
  VIRTUAL_ENV="$(cd "$(dirname "${PYTHON}")/.." && pwd)" \
    uv pip install --python "${PYTHON}" --no-build-isolation -e .
else
  "${PYTHON}" -m pip install "scikit-build-core>=0.10" "cmake>=3.24" ninja "pybind11>=2.13"
  "${PYTHON}" -m pip install --no-build-isolation -e .
fi
