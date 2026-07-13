#!/bin/bash
# Install Boost on macOS via Homebrew. No sudo / no system symlinks required.
# install.sh picks up the include path via `brew --prefix boost`.

set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh/ and re-run." >&2
  exit 1
fi

if ! brew list boost >/dev/null 2>&1; then
  echo "Installing boost via Homebrew..."
  brew install boost
fi

BOOST_PREFIX="$(brew --prefix boost)"
BOOST_INCLUDE="${BOOST_PREFIX}/include"

if [ ! -d "${BOOST_INCLUDE}/boost" ]; then
  echo "Expected boost headers at ${BOOST_INCLUDE}/boost but directory not found." >&2
  exit 1
fi

echo "Boost headers are at: ${BOOST_INCLUDE}"
echo "install.sh will pick this up automatically via \`brew --prefix boost\`."
