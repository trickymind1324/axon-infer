#!/usr/bin/env bash
# Runs every gate on the local machine: format check, every preset's build and tests,
# clang-tidy, integration tests when models are present, and a benchmark smoke run. Exits
# non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

PRESETS="${PRESETS:-debug release asan-ubsan tsan}"
LLVM_BIN=/opt/homebrew/opt/llvm/bin
if [ -d "$LLVM_BIN" ]; then
  export PATH="$LLVM_BIN:$PATH"
fi

step() { printf '\n==> %s\n' "$*"; }

step "clang-format"
scripts/format.sh --check

for preset in $PRESETS; do
  step "configure $preset"
  cmake --preset "$preset" >/dev/null
  step "build $preset"
  cmake --build --preset "$preset"
  step "test $preset"
  ctest --preset "$preset"
done

if [ "${SKIP_TIDY:-0}" != "1" ]; then
  step "clang-tidy"
  TIDY_ARGS=()
  if [ -x "$LLVM_BIN/clang-tidy" ]; then
    TIDY_ARGS+=("-DAXON_CLANG_TIDY_EXE=$LLVM_BIN/clang-tidy")
  fi
  cmake --preset tidy "${TIDY_ARGS[@]}" >/dev/null
  cmake --build --preset tidy
fi

# ctest runs each test from the build tree, so the model and fixture paths must be absolute.
if [ -d "${AXON_MODELS_DIR:-models}" ] && ls "${AXON_MODELS_DIR:-models}"/*.gguf >/dev/null 2>&1; then
  step "integration tests (models found)"
  MODELS_ABS="$(cd "${AXON_MODELS_DIR:-models}" && pwd)"
  LOCAL_ABS=""
  if [ -d tests/reference/local ]; then
    LOCAL_ABS="$(cd tests/reference/local && pwd)"
  fi
  AXON_MODELS_DIR="$MODELS_ABS" AXON_LOCAL_FIXTURES="$LOCAL_ABS" \
    ctest --preset release -L integration --output-on-failure
fi

step "benchmark smoke"
./build/release/bench/bench_kernels --benchmark_min_time=0.01s
./build/release/apps/axon/axon bench -m tests/reference/fixtures/gguf/synthetic-2l-q8_0.gguf --prompt-tokens 32 -n 8 --runs 2 --ctx 128 --json > /dev/null
"${PYTHON:-python3}" bench/render_tables.py --self-test

step "all checks passed"
