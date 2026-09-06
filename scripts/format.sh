#!/usr/bin/env bash
# Formats every project source with clang-format. Pass --check to only verify.
set -euo pipefail
cd "$(dirname "$0")/.."

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
if ! command -v "$CLANG_FORMAT" >/dev/null 2>&1 && [ -x /opt/homebrew/opt/llvm/bin/clang-format ]; then
  CLANG_FORMAT=/opt/homebrew/opt/llvm/bin/clang-format
fi

find_sources() {
  find include src apps bench tests -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.h.in' \) -print0 | sort -z
}

if [ "${1:-}" = "--check" ]; then
  find_sources | xargs -0 "$CLANG_FORMAT" --dry-run --Werror
else
  find_sources | xargs -0 "$CLANG_FORMAT" -i
fi
