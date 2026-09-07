#!/usr/bin/env bash
# Runs the same gates as the CI workflow on the local machine: format check, every preset's
# build and tests, clang-tidy, and a benchmark smoke run. Exits non-zero on the first failure.
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

if [ -d "${AXON_MODELS_DIR:-models}" ] && ls "${AXON_MODELS_DIR:-models}"/*.gguf >/dev/null 2>&1; then
  step "integration tests (models found)"
  AXON_MODELS_DIR="${AXON_MODELS_DIR:-models}" ctest --preset release -L integration
fi

step "benchmark smoke"
./build/release/bench/bench_kernels --benchmark_min_time=0.01s

step "all checks passed"
