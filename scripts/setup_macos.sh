#!/usr/bin/env bash
# Installs the local toolchain and the Python environment used to generate test fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."

brew install cmake ninja llvm

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy torch transformers gguf safetensors sentencepiece
echo "done: cmake, ninja, llvm installed; .venv ready"
