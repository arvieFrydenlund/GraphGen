#!/bin/bash
set -euo pipefail

# Resolve the Python interpreter. Prefer the project's uv/venv Python
# (which has pybind11 installed and matches pyproject.toml's requires-python),
# then fall back to whatever `python3` is on PATH. Callers can override with
# `PYTHON=/path/to/python bash install.sh`.
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

# Derive the extension suffix (e.g. .cpython-310-darwin.so) from sysconfig
# rather than python3-config, since uv-managed envs don't ship python3-config.
EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYBIND11_INCLUDES="$("${PYTHON}" -m pybind11 --includes)"

# Resolve the Boost include dir. On macOS we install Boost via Homebrew
# (see install_mac_boost.sh) and pick it up here without needing sudo /
# system symlinks. On Linux fall back to the traditional /usr/include path.
if [[ "$(uname -s)" == "Darwin" ]] \
    && command -v brew >/dev/null 2>&1 \
    && brew list boost >/dev/null 2>&1; then
  BOOST_INCLUDE_DIR="$(brew --prefix boost)/include"
else
  BOOST_INCLUDE_DIR="/usr/include"
fi

# On macOS, `g++` is actually clang++, which (a) refuses to accept header
# files as compilation inputs alongside a single `-o` output, and (b) needs
# `-undefined dynamic_lookup` for pybind11 modules so Python's symbols are
# resolved at import time instead of at link time. On real g++ (Linux),
# keep the original invocation so behavior there is unchanged.
if g++ --version 2>/dev/null | grep -qi clang; then
  g++ -std=c++20 -O3 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -undefined dynamic_lookup -Wno-sign-compare -Wunused-variable \
    -fPIC ${PYBIND11_INCLUDES} \
    -I"${BOOST_INCLUDE_DIR}" \
    -I. generator.cpp -o "generator${EXT_SUFFIX}"
else
  g++ -std=c++20 -Ofast -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -Wno-sign-compare -Wunused-variable \
    -fPIC ${PYBIND11_INCLUDES} \
    -I"${BOOST_INCLUDE_DIR}" \
    -I. undirected_graphs.h directed_graphs.h utils.h dictionaries.h matrix.h args.h graph_wrapper.h graph_tokenizer.h tasks.h scratch_pads.h instance.h generator.cpp -o "generator${EXT_SUFFIX}"
fi

"${PYTHON}" setup.py build
"${PYTHON}" setup.py install
