#!/usr/bin/env bash
# Downloads the GGUF models used by the integration tests and benchmarks, verifying SHA256.
#
#   scripts/download_models.sh                # the default set (SmolLM2-135M in F16, Q8_0, Q4_0)
#   scripts/download_models.sh --all          # every pinned model (about 7.5 GB)
#   scripts/download_models.sh llama-3.2-1b-q4_0 qwen2.5-0.5b-q8_0
#   scripts/download_models.sh --list
#
# Files land in ./models (override with --dir or AXON_MODELS_DIR). Existing files with the
# right hash are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="${AXON_MODELS_DIR:-models}"
BASE="https://huggingface.co"

# name|repo|file|sha256
MODELS=(
  "smollm2-135m-f16|bartowski/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct-f16.gguf|f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
  "smollm2-135m-q8_0|bartowski/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct-Q8_0.gguf|5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
  "smollm2-135m-q4_0|bartowski/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct-Q4_0.gguf|bcc3af2849ad6095af57e9b5cd43775256efdc66e306acb529172f92d0c04b03"
  "smollm2-1.7b-q4_0|bartowski/SmolLM2-1.7B-Instruct-GGUF|SmolLM2-1.7B-Instruct-Q4_0.gguf|99b866e77adee858930244229afa833dbe3372a7bf122905ea506119c6911ae8"
  "tinyllama-1.1b-f16|mradermacher/TinyLlama-1.1B-Chat-v1.0-GGUF|TinyLlama-1.1B-Chat-v1.0.f16.gguf|f83b5b067440dc582f8a9c217c3a17c15234a5c7ce1899f54d3a9f0f2007245a"
  "tinyllama-1.1b-q4_0|TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF|tinyllama-1.1b-chat-v1.0.Q4_0.gguf|da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"
  "qwen2.5-0.5b-q8_0|Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q8_0.gguf|ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
  "qwen2.5-0.5b-q4_0|Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q4_0.gguf|7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed"
  "qwen2.5-1.5b-q4_0|Qwen/Qwen2.5-1.5B-Instruct-GGUF|qwen2.5-1.5b-instruct-q4_0.gguf|dcd819ff094852c38faba6873d8ff0c9d51eadb2844539e52042ae5d647bbfdb"
  "llama-3.2-1b-q4_0|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_0.gguf|fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8"
  "llama-3.2-1b-q8_0|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q8_0.gguf|432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
)
DEFAULT="smollm2-135m-f16 smollm2-135m-q8_0 smollm2-135m-q4_0"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

list_models() {
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo file _ <<<"$entry"
    printf '  %-22s %s/%s\n' "$name" "$repo" "$file"
  done
}

SELECTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --all) for e in "${MODELS[@]}"; do SELECTED+=("${e%%|*}"); done ;;
    --list) list_models; exit 0 ;;
    --dir) DIR="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) SELECTED+=("$1") ;;
  esac
  shift
done
if [ ${#SELECTED[@]} -eq 0 ]; then
  # shellcheck disable=SC2206
  SELECTED=($DEFAULT)
fi

mkdir -p "$DIR"
status=0
for name in "${SELECTED[@]}"; do
  found=""
  for entry in "${MODELS[@]}"; do
    if [ "${entry%%|*}" = "$name" ]; then found="$entry"; break; fi
  done
  if [ -z "$found" ]; then
    echo "unknown model '$name' (see --list)" >&2
    status=1
    continue
  fi
  IFS='|' read -r _ repo file sha <<<"$found"
  target="$DIR/$file"
  if [ -f "$target" ] && [ "$(sha256_of "$target")" = "$sha" ]; then
    echo "ok       $file"
    continue
  fi
  echo "download $file"
  curl -L --fail --progress-bar -o "$target.part" "$BASE/$repo/resolve/main/$file"
  actual="$(sha256_of "$target.part")"
  if [ "$actual" != "$sha" ]; then
    echo "checksum mismatch for $file: expected $sha, found $actual" >&2
    rm -f "$target.part"
    status=1
    continue
  fi
  mv "$target.part" "$target"
  echo "verified $file"
done
exit $status
