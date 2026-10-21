#!/usr/bin/env bash
# Runs `axon bench` for one model at one or more thread counts and writes one JSON file per
# thread count under bench/results/, wrapped with the host metadata the methodology requires:
# host name, CPU, cores, RAM, OS, compiler, git commit, model file, size and SHA256, and the
# date. PEAK_GB_S is the host's memory bandwidth as published by its maker, recorded so the
# renderer can show achieved fractions without hard-coding a machine.
#
#   STEP=06a-threads HOST=m1 PEAK_GB_S=68.25 bench/bench_e2e.sh models/SmolLM2-135M-Instruct-Q8_0.gguf 1 2 4 8
#
# Extra arguments for axon bench go in BENCH_ARGS, for example BENCH_ARGS="--runs 3".
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: STEP=<step> HOST=<host> bench/bench_e2e.sh <model.gguf> <threads>...}"
shift
THREAD_COUNTS=("$@")
[ "${#THREAD_COUNTS[@]}" -gt 0 ] || THREAD_COUNTS=(0)
STEP="${STEP:?set STEP, for example 06a-threads}"
HOST="${HOST:-$(hostname -s)}"
PEAK_GB_S="${PEAK_GB_S:-0}"
AXON="${AXON:-build/release/apps/axon/axon}"
PYTHON="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
[ -x "$AXON" ] || { echo "missing $AXON, build the release preset first" >&2; exit 1; }
mkdir -p bench/results

VERSION="$("$AXON" --version)"
COMPILER="$(printf '%s\n' "$VERSION" | sed -n 's/^compiler: //p')"
CPU="$(printf '%s\n' "$VERSION" | sed -n 's/^cpu: //p')"
COMMIT="$(git rev-parse --short=12 HEAD)"
DIRTY="$(git diff --quiet && echo False || echo True)"
case "$(uname -s)" in
  Darwin) OS="macOS $(sw_vers -productVersion) ($(uname -m))"; RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); SHA="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)";;
  *) OS="$(uname -sr) ($(uname -m))"; RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)";;
esac
MODEL_BYTES="$(wc -c < "$MODEL" | tr -d ' ')"
MODEL_NAME="$(basename "$MODEL" .gguf)"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for threads in "${THREAD_COUNTS[@]}"; do
  echo "==> $MODEL_NAME, $threads threads"
  # shellcheck disable=SC2086
  RESULT="$("$AXON" bench -m "$MODEL" -t "$threads" --json ${BENCH_ARGS:-})"
  SUFFIX="$(printf '%s' "$RESULT" | "$PYTHON" -c 'import json, sys; r = json.load(sys.stdin); b = r.get("batch", 1); d = r.get("draft", ""); print((f"-b{b}" if b > 1 else "") + (f"-spec{r.get(\"draft_tokens\", 0)}" if d else ""))')"
  OUT="bench/results/${HOST}-${STEP}-${MODEL_NAME}-t${threads}${SUFFIX}.json"
  "$PYTHON" - "$RESULT" "$OUT" <<PY
import json, sys
result = json.loads(sys.argv[1])
result["description"] = result.pop("model")  # the model key below holds the file metadata
record = {
    "step": "$STEP", "date": "$DATE", "host": "$HOST", "cpu": "$CPU", "ram_gb": $RAM_GB,
    "os": "$OS", "compiler": "$COMPILER", "commit": "$COMMIT", "dirty": $DIRTY,
    "peak_gb_s": $PEAK_GB_S,
    "model": {"file": "$MODEL_NAME", "bytes": $MODEL_BYTES, "sha256": "$SHA"},
}
record.update(result)
with open(sys.argv[2], "w") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
print(f"    prefill {result['prefill_tok_s']['median']:.2f} tok/s, decode {result['decode_tok_s']['median']:.2f} tok/s, "
      f"{result['weight_gb_s']:.2f} GB/s -> {sys.argv[2]}")
PY
done
