#!/usr/bin/env bash
# Builds the release preset from a clean build directory, runs the unit tests on it, and packs
# the four binaries with the LICENSE and a note naming the commit into one tarball with a
# SHA256 file next to it. Refuses a dirty tree, so a tarball always matches a commit.
#
#   scripts/package_release.sh                 # writes dist/axon-infer-<version>-<os>-<arch>.tar.gz
#   scripts/package_release.sh --no-tests      # skip the unit tests (the Docker stage runs them)
#
# The Linux tarball comes out of the Dockerfile's build stage with the same script, see the
# README's development section.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TESTS=1
[ "${1:-}" = "--no-tests" ] && RUN_TESTS=0

# Inside the Docker build there is no .git, and the commit comes in as AXON_COMMIT.
if [ -d .git ]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "the tree has uncommitted changes, commit or stash them first" >&2
    exit 1
  fi
  COMMIT="$(git rev-parse --short=12 HEAD)"
else
  COMMIT="${AXON_COMMIT:?set AXON_COMMIT when packaging outside a checkout}"
fi
VERSION="$(sed -n 's/^  VERSION \([0-9][0-9.]*\)$/\1/p' CMakeLists.txt | head -1)"
[ -n "$VERSION" ] || { echo "cannot read the version from CMakeLists.txt" >&2; exit 1; }
case "$(uname -s)" in
  Darwin) OS=macos;;
  Linux) OS=linux;;
  *) OS="$(uname -s | tr '[:upper:]' '[:lower:]')";;
esac
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] && ARCH=arm64
NAME="axon-infer-${VERSION}-${OS}-${ARCH}"

BUILD="build/package"
rm -rf "$BUILD"
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DAXON_BUILD_BENCH=OFF \
  -DAXON_BUILD_TESTS="$([ "$RUN_TESTS" = 1 ] && echo ON || echo OFF)" > "$BUILD.configure.log" 2>&1 \
  || { tail -20 "$BUILD.configure.log" >&2; exit 1; }
cmake --build "$BUILD" > "$BUILD.build.log" 2>&1 || { tail -20 "$BUILD.build.log" >&2; exit 1; }
if [ "$RUN_TESTS" = 1 ]; then
  "$BUILD/tests/axon_tests" > "$BUILD.tests.log" 2>&1 || { tail -20 "$BUILD.tests.log" >&2; exit 1; }
fi

STAGE="dist/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$BUILD/apps/axon/axon" "$BUILD/apps/axon-server/axon-server" \
   "$BUILD/apps/axon-perplexity/axon-perplexity" "$BUILD/apps/axon-dump/axon-dump" "$STAGE/"
cp LICENSE "$STAGE/"
{
  echo "axon-infer $VERSION"
  echo "commit $COMMIT"
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ) on $OS $ARCH"
  echo
  echo "axon            chat, generate, bench and tokenize with a GGUF model"
  echo "axon-server     the OpenAI chat completions API over one model"
  echo "axon-perplexity perplexity of a model over a text"
  echo "axon-dump       the metadata and tensors of a GGUF file"
  echo
  echo "Models are not included. scripts/download_models.sh in the repository fetches the pinned files."
} > "$STAGE/README.txt"
tar -C dist -czf "dist/$NAME.tar.gz" "$NAME"
( cd dist && { command -v sha256sum >/dev/null 2>&1 && sha256sum "$NAME.tar.gz" || shasum -a 256 "$NAME.tar.gz"; } > "$NAME.tar.gz.sha256" )
rm -rf "$STAGE"
echo "dist/$NAME.tar.gz"
cat "dist/$NAME.tar.gz.sha256"
