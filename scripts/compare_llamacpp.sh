#!/usr/bin/env bash
# Measures llama.cpp on the same files, prompt length, generation length and repetition count
# as `axon bench`, and writes one JSON record per thread count under bench/results/ with the
# step `llamacpp`, so bench/render_tables.py can put the two engines side by side.
#
#   HOST=m1 scripts/compare_llamacpp.sh models/Llama-3.2-1B-Instruct-Q4_0.gguf 4 8
#
# llama.cpp is cloned into the gitignored third_party/llama.cpp at the pinned commit and built
# CPU-only (no Metal, no BLAS, no Accelerate) in release mode. The build runs before any model
# is loaded. `llama-bench -p 512 -n 128 -r 5` reports the prompt pass (pp512) and the
# generation pass (tg128) with one sample per repetition, and the record keeps the medians
# and the best of those samples, like the engine's records do. Two differences from the engine
# are recorded rather than adjusted: tg128 generates from an empty context where the engine
# decodes after the 512-token prompt, and llama.cpp repacks Q4_0 weights into its interleaved
# arm64 layout at load.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: HOST=<host> scripts/compare_llamacpp.sh <model.gguf> <threads>... (or --build-only)}"
shift
BUILD_ONLY=0
[ "$MODEL" = "--build-only" ] && BUILD_ONLY=1
THREAD_COUNTS=("$@")
[ "${#THREAD_COUNTS[@]}" -gt 0 ] || THREAD_COUNTS=(0)
HOST="${HOST:-$(hostname -s)}"
STEP="llamacpp"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GENERATE="${GENERATE:-128}"
RUNS="${RUNS:-5}"
# The commit every published llama.cpp number comes from. Override to measure another one.
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-}"
LLAMACPP_DIR="${LLAMACPP_DIR:-third_party/llama.cpp}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
JOBS="${JOBS:-4}"
AXON="${AXON:-build/release/apps/axon/axon}"
PYTHON="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
[ -x "$AXON" ] || { echo "missing $AXON, build the release preset first (its --version supplies the host metadata)" >&2; exit 1; }
[ "$BUILD_ONLY" = 1 ] || [ -f "$MODEL" ] || { echo "missing model $MODEL" >&2; exit 1; }
mkdir -p bench/results third_party

if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  echo "==> cloning llama.cpp into $LLAMACPP_DIR"
  git clone --quiet "$LLAMACPP_REPO" "$LLAMACPP_DIR"
fi
if [ -n "$LLAMACPP_COMMIT" ]; then
  git -C "$LLAMACPP_DIR" fetch --quiet origin "$LLAMACPP_COMMIT" 2>/dev/null || git -C "$LLAMACPP_DIR" fetch --quiet origin
  git -C "$LLAMACPP_DIR" checkout --quiet "$LLAMACPP_COMMIT"
else
  echo "LLAMACPP_COMMIT is not set, measuring the checked-out commit and printing it below" >&2
fi
RESOLVED="$(git -C "$LLAMACPP_DIR" rev-parse HEAD)"
BUILD_NUMBER="$(git -C "$LLAMACPP_DIR" rev-list --count HEAD)"
FLAGS="-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release"
BENCH="$LLAMACPP_DIR/build/bin/llama-bench"
if [ ! -x "$BENCH" ] || [ "$(cat "$LLAMACPP_DIR/build/.axon-commit" 2>/dev/null)" != "$RESOLVED" ]; then
  echo "==> building llama-bench at $RESOLVED (CPU only)"
  # shellcheck disable=SC2086
  cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" $FLAGS > "$LLAMACPP_DIR/build-configure.log" 2>&1 || { tail -20 "$LLAMACPP_DIR/build-configure.log" >&2; exit 1; }
  cmake --build "$LLAMACPP_DIR/build" --target llama-bench -j "$JOBS" > "$LLAMACPP_DIR/build.log" 2>&1 || { tail -20 "$LLAMACPP_DIR/build.log" >&2; exit 1; }
  echo "$RESOLVED" > "$LLAMACPP_DIR/build/.axon-commit"
fi
echo "llama.cpp commit $RESOLVED (build number $BUILD_NUMBER)"
[ "$BUILD_ONLY" = 1 ] && exit 0

VERSION="$("$AXON" --version)"
CPU="$(printf '%s\n' "$VERSION" | sed -n 's/^cpu: //p')"
COMMIT="$(git rev-parse --short=12 HEAD)"
case "$(uname -s)" in
  Darwin) OS="macOS $(sw_vers -productVersion) ($(uname -m))"; RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); SHA="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)";;
  *) OS="$(uname -sr) ($(uname -m))"; RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)";;
esac
MODEL_BYTES="$(wc -c < "$MODEL" | tr -d ' ')"
MODEL_NAME="$(basename "$MODEL" .gguf)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for threads in "${THREAD_COUNTS[@]}"; do
  echo "==> llama.cpp, $MODEL_NAME, $threads threads"
  RESULT="$("$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GENERATE" -r "$RUNS" -t "$threads" -o json 2>/dev/null)"
  OUT="bench/results/${HOST}-${STEP}-${MODEL_NAME}-t${threads}.json"
  "$PYTHON" - "$RESULT" "$OUT" "$threads" <<PY
import json, statistics, sys
tests = json.loads(sys.argv[1])
if not isinstance(tests, list) or len(tests) < 2:
    sys.exit("unexpected llama-bench output: " + sys.argv[1][:200])
def rates(test):
    tokens = test["n_prompt"] + test["n_gen"]
    samples = [tokens / (ns / 1e9) for ns in test["samples_ns"]]
    return {"median": statistics.median(samples), "best": max(samples), "samples": samples}
prompt = next(t for t in tests if t["n_prompt"] == $PROMPT_TOKENS and t["n_gen"] == 0)
gen = next(t for t in tests if t["n_gen"] == $GENERATE and t["n_prompt"] == 0)
record = {
    "step": "$STEP", "date": "$DATE", "host": "$HOST", "cpu": "$CPU", "ram_gb": $RAM_GB,
    "os": "$OS", "commit": "$COMMIT",
    "model": {"file": "$MODEL_NAME", "bytes": $MODEL_BYTES, "sha256": "$SHA"},
    "llamacpp": {"commit": "$RESOLVED", "build_number": $BUILD_NUMBER, "flags": "$FLAGS",
                 "build_commit": prompt.get("build_commit"), "cpu_info": prompt.get("cpu_info"),
                 "backends": prompt.get("backends")},
    "threads": int(sys.argv[3]), "prompt_tokens": $PROMPT_TOKENS, "generated_tokens": $GENERATE,
    "runs": $RUNS,
    "prefill_tok_s": rates(prompt), "decode_tok_s": rates(gen),
}
with open(sys.argv[2], "w") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
print(f"    pp{$PROMPT_TOKENS} {record['prefill_tok_s']['median']:.2f} tok/s, tg{$GENERATE} {record['decode_tok_s']['median']:.2f} tok/s -> {sys.argv[2]}")
PY
done
