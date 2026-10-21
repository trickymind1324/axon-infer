# Contributing

Thanks for looking at **axon-infer**. The repository is small enough to read end to end, and it keeps a few rules that make its numbers trustworthy. This page says what they are and how to work within them.

## Build and check

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
scripts/check_all.sh
```

`scripts/check_all.sh` is the gate for every change. It runs the format check, the debug, release, ASan+UBSan and TSan builds with their tests, clang-tidy, the integration tests when model files are present, and a benchmark smoke run. A change is ready when the script passes on your machine. There is no hosted CI, so the script is the whole story.

Model-backed tests are opt-in. `scripts/download_models.sh` fetches the pinned files with their checksums, and `AXON_MODELS_DIR=$PWD/models scripts/check_all.sh` runs the integration tests against them. The default set is the three SmolLM2-135M files, and `--list` shows the rest.

## The locked tests

Committed tests, accuracy gates, fixtures and benchmark rows are locked. A red run is a finding, not a nuisance: investigate it and report what you found, and never change a gate, a fixture or a tolerance to make a run green. New tolerances are set by measuring first and locking afterwards with a margin, and the measurement and its date go in a comment next to the number. If a locked expectation turns out to be wrong, the change comes with the evidence that shows why, as its own commit.

The same applies to numbers in the README. Every benchmark row comes from a JSON record under `bench/results/` written by `bench/bench_e2e.sh` or `scripts/compare_llamacpp.sh`, and the tables are rendered from those records by `bench/render_tables.py`. Records are never edited by hand. If a number looks wrong, re-run it.

## Identities the code relies on

A few properties are asserted as equality in the tests, and changes must keep them:

- The logits do not depend on the thread count. No partial sum crosses a thread.
- Cached decoding equals the cache-free forward bit for bit with the F32 cache.
- The multi-row kernels equal the single-row kernels row by row, so prefill, batched decode and speculative scoring all compute what single-step decode computes.
- The integer NEON kernels are bit-identical to the scalar reference. The float NEON kernels differ by reduction order within a measured, locked tolerance.

If a change cannot keep one of these, say so in the proposal and explain what replaces it.

## Adding a kernel

Declare it in `src/kernels/kernels.h` and `src/kernels/ref/ref.h`, implement the scalar version under `src/kernels/ref/`, register it in `src/kernels/kernels.cpp`, add its reference to the fixture generator under `tests/reference/` and regenerate, then write a fixture test and at least one property test under `tests/unit/`. A NEON version goes under `src/kernels/neon/` with its own test against the scalar one, as the existing files show.

## Style

The code is C++20 formatted by clang-format and checked by clang-tidy with the configuration in the repository. Comments say why, not what. Commit messages are plain sentences in the imperative that read like a person wrote them, and the README is updated in the same change when behaviour or numbers change.

## Reporting

Use the issue templates. A bug report wants the `--version` output, the model file with its SHA256, the command and what it printed. A number report wants the JSON record the bench script wrote, so the row can be rendered next to the others.
