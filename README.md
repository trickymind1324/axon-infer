# axon-infer

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![C++20](https://img.shields.io/badge/C%2B%2B-20-informational) ![Platform](https://img.shields.io/badge/platform-macOS%20arm64-lightgrey)

LLM inference engine in C++20, built from scratch. It loads GGUF models from the LLaMA family and runs them on CPU with memory-mapped weights, a KV cache, a chat template renderer and a seeded sampler. No ggml, BLAS, OpenMP or other ML runtime. Every component is checked against a reference implementation, and every optimization is benchmarked so you can see what it buys.

## Contents

- [Overview](#overview)
- [Features](#features)
- [Supported models and formats](#supported-models-and-formats)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Correctness](#correctness)
- [Benchmarks](#benchmarks)
- [Design notes](#design-notes)
- [Development](#development)
- [License](#license)

## Overview

LLM inference is memory-bandwidth-bound on most hardware, not compute-bound. Production engines make that hard to see because they are hundreds of thousands of lines. axon-infer is a compact, tested engine where the whole path from file bytes to sampled token is readable, and where the benchmark tables double as the explanation.

The scope is the LLaMA block: RMSNorm, rotary embeddings, grouped-query attention and a SwiGLU feed-forward. Training, multimodal models and other architectures are out of scope, and so is beating llama.cpp. The goal is a clean, measured, understandable engine.

**Current limitations.** The kernels are scalar and single-threaded, so throughput is a few tokens per second on a 135M model. Quantized weights (Q8_0 and Q4_0) load and can be inspected but do not run yet. There is no HTTP server yet.

## Features

- **GGUF loader** over a read-only memory mapping, validating every offset and length before any accessor can be called, fuzzed with a deterministic corruption replay and a libFuzzer target.
- **Tokenizers** built entirely from GGUF metadata: byte-level BPE with the GPT-2, Llama 3 and Qwen2 pre-tokenizer patterns, and SentencePiece BPE with byte fallback, plus NFC normalization and a streaming decoder that never emits a partial UTF-8 sequence.
- **Forward pass** on a table of kernels with a scalar reference for each, activations from one arena sized at load, and no heap allocation per token.
- **KV cache** with prefill, single-token decode and truncation, bit-identical to the cache-free forward.
- **Sampler** with temperature, top-k, top-p, min-p and a repetition penalty, on a seeded generator that reproduces across platforms.
- **Chat** through the model's own Jinja template, rendered by a small interpreter with per-family fallbacks, reusing the cached prefix of the conversation across turns.
- **Tools**: `axon` (chat, generate, tokenize), `axon-dump` for inspecting any GGUF file, benchmarks for kernels and the forward pass.

## Supported models and formats

Architectures read from GGUF metadata:

| Architecture | Config | Tokenizer | Forward pass | Verified against Hugging Face |
|---|---|---|---|---|
| `llama` (Llama 2, Llama 3.x, SmolLM2, TinyLlama) | ✅ | ✅ | ✅ | SmolLM2-135M: logits, greedy output, chat template |
| `qwen2` (Qwen2.5) | ✅ | ✅ | ✅ (Q/K/V bias, NeoX rotary layout) | tokenizer and chat template |

Weight formats:

| Format | Load and inspect | Run |
|---|---|---|
| F32 | ✅ | ✅ |
| F16 | ✅ | ✅ |
| BF16 | ✅ | ✅ |
| Q8_0 | ✅ | not yet |
| Q4_0 | ✅ | not yet |
| K-quants and others | ✅ (dump only) | no |

Models exercised by the test suite: SmolLM2-135M-Instruct (F16, Q8_0, Q4_0), TinyLlama-1.1B-Chat (Q4_0), Llama-3.2-1B-Instruct (Q4_0) and Qwen2.5-0.5B-Instruct (Q4_0). Chat runs on SmolLM2 today, and the other three join once quantized kernels land.

Platforms: developed and benchmarked on macOS arm64 (Apple M1). The code is portable C++20 with no platform-specific kernels yet, and Linux x86-64 builds are untested.

## Architecture

```
GGUF file ──mmap──▶ GGUFFile ──▶ weight binding ──▶ LlamaModel
                                                       │
prompt ──▶ Tokenizer ──▶ ids ──▶ prefill ──▶ KV cache ─┤
                                                       ▼
                       Sampler ◀── logits ◀── decode (one token per step)
                          │
                          ▼
                    Tokenizer.decode ──▶ streamed text
```

Layers underneath: `core` (arena allocator, thread pool, GGUF reader, dtypes), `kernels` (the scalar reference table), `model` (LLaMA block, KV cache), `tokenizer`, `sampling`, `chat` (Jinja interpreter and templates), `engine` (generation loop).

Design rules: weights are memory-mapped and never copied. Activations come from one arena sized at load. The decode loop never allocates on the heap. Every kernel has a scalar reference and a test against it before any optimized version exists.

## Quickstart

Requirements: CMake 3.24+, Ninja, a C++20 compiler (clang 17+, gcc 13+, or Apple clang 15+).

```sh
cmake --preset release
cmake --build --preset release
./build/release/apps/axon/axon --version

scripts/download_models.sh                     # SmolLM2-135M in F16, Q8_0 and Q4_0, checksums verified
./build/release/apps/axon/axon chat -m models/SmolLM2-135M-Instruct-f16.gguf
```

## Usage

### Chat

`axon chat` holds a multi-turn conversation using the model's own chat template. Each turn reuses the cached prefix of the conversation, so a follow-up question costs only its new tokens:

```
$ ./build/release/apps/axon/axon chat -m models/SmolLM2-135M-Instruct-f16.gguf --temp 0 --stats
> What is the capital of France?
The capital of France is Paris. It's a city that has been a beacon for art, culture, and innovation since ancient times.
[prompt 37 tokens (0 reused), first token in 9.13 s, decode 40 tokens in 9.65 s (4.15 tok/s), cache 76 positions (3.3 MB of 90.0 MB), peak rss 273 MB, seed 15111573243193277517]
> And of Germany?
Germany also has its own capital city, Berlin. However, it's not the same as Paris in France or London in England.
[prompt 92 tokens (76 reused), first token in 3.92 s, decode 40 tokens in 9.78 s (4.09 tok/s), cache 131 positions (5.8 MB of 90.0 MB), peak rss 273 MB, seed 15111573243193277517]
```

`/clear` starts a new conversation, `/stats` repeats the last turn's numbers, and `/exit` quits. Sampling is controlled with `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty` and `--seed`. `--system` sets a system prompt and `--ctx` the cache length.

### Generate

`axon generate` completes a prompt. It is greedy by default so its output matches Hugging Face token for token, and it takes the same sampling flags as `chat`:

```
$ ./build/release/apps/axon/axon generate -m models/SmolLM2-135M-Instruct-f16.gguf -p "The capital of France is" -n 20
model: Smollm2 135M 8k Lc100K Mix1 Ep2 (llama layers 30 embd 576 heads 9 kv 3 ff 1536 vocab 49152 ctx 8192 rope normal 100000 tied)
prompt: 5 tokens
The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace
prompt 5 tokens (0 reused), first token in 1.24 s, decode 20 tokens in 4.65 s (4.30 tok/s), cache 24 positions (1.1 MB of 90.0 MB), peak rss 273 MB, seed 15111573250111174978
```

`--ctx` sets the cache length (default 2048, clamped to the model's training context). `--no-cache` recomputes the whole sequence each step, which is the reference path and the one that can dump activations with `--dump-layer N --dump-dir D`.

### Tokenize

`axon tokenize` encodes text with a model's tokenizer and shows every token:

```
$ ./build/release/apps/axon/axon tokenize -m models/SmolLM2-135M-Instruct-Q4_0.gguf "Hello world! <|im_end|>"
model: Smollm2 135M 8k Lc100K Mix1 Ep2 (gpt2, pre smollm), vocab 49152
[  19556] "Hello"
[    905] " world"
[     17] "!"
[    216] " "
[      2] "<|im_end|>"  (special)
5 tokens for 23 bytes
```

### Inspect a GGUF file

`axon-dump` prints the header, every metadata key, the tensor table and a per-dtype byte summary. It reads any GGUF v3 file, including K-quant models the engine cannot run, and `--json` gives the same data as a document:

```
$ ./build/release/apps/axon-dump/axon-dump models/SmolLM2-135M-Instruct-Q8_0.gguf --no-arrays
version 3  alignment 32  tensors 272  metadata 37  size 138.1 MB  data offset 1785952
architecture llama  name "Smollm2 135M 8k Lc100K Mix1 Ep2"
...
summary
  F32         61 tensors        140544 bytes   137.2 KB    0.1%
  Q8_0       211 tensors     142884864 bytes   136.3 MB   99.9%
  total      272 tensors     143025408 bytes   136.4 MB
```

## Correctness

Every component is checked against an independent reference. The small fixtures are committed, so the regular test suite needs no download. Tests against real model files are opt-in.

| Component | Reference | Check |
|---|---|---|
| GGUF loader | the official `gguf` Python package | tensor tables of downloaded models match a committed listing, and 5000 deterministic corruptions per fixture never crash |
| Tokenizers | Hugging Face `tokenizers` | exact id match on a 170-line corpus for SmolLM2, TinyLlama, Llama 3.2 and Qwen2.5, covering contractions, numbers, whitespace, code, emoji and a dozen scripts |
| Kernels | NumPy in float64, with PyTorch pinning the RoPE and attention conventions | fixture tests per kernel plus property tests |
| Forward pass | NumPy on a synthetic two-layer model, and Hugging Face `LlamaForCausalLM` loaded from the same GGUF file for SmolLM2-135M | argmax matches at every position of ten prompts, logits agree to better than 5e-5 on a range of about 46, and 20-token greedy continuations match |
| KV cache | the cache-free forward | cached decoding is bit-identical at every position, however the tokens are split between prefill chunks and single steps |
| Chat templates | Hugging Face `apply_chat_template` | twelve renders across four model families match byte for byte |
| Sampler | closed-form expectations | top-k and top-p bounds, temperature zero equals argmax, seeded reproducibility, and draw frequencies within two percent |

Every preset runs the unit tests: debug, release, AddressSanitizer with UndefinedBehaviorSanitizer, and ThreadSanitizer. clang-tidy runs with warnings as errors.

## Benchmarks

All numbers are single-threaded on an Apple M1 with the scalar kernels, measured by the binaries under `bench/`. Each row is the median of repeated runs after a warm-up.

End-to-end on SmolLM2-135M with F16 weights:

| Configuration | Prefill tok/s | Generate tok/s |
|---|---|---|
| Scalar, 1 thread, no cache | 4.1 | 0.28 |
| Scalar, 1 thread, KV cache | 4.1 | 4.3 |

The first row is the deliberately naive baseline: every generated token recomputes the whole sequence. The KV cache turns generation into one forward pass per token, so decode runs at the same rate as prefill. Every later optimization is measured against these rows.

Decode cost per token is nearly flat in context depth on the scalar path, because the attention over cached positions is small next to the weight matmuls:

| Cached positions | ms per decoded token |
|---|---|
| 32 | 252 |
| 512 | 248 |
| 2048 | 266 |

Scalar matmul, the kernel that dominates decode:

| Weights | Shape (out x in) | GFLOP/s | GB/s |
|---|---|---|---|
| F32 | 1536 x 576 | 2.6 | 5.2 |
| F32 | 2048 x 2048 | 2.1 | 4.3 |
| F32 | 8192 x 2048 | 2.2 | 4.4 |
| F16 | 1536 x 576 | 1.1 | 1.1 |
| F16 | 2048 x 2048 | 1.1 | 1.1 |
| F16 | 8192 x 2048 | 1.1 | 1.1 |

The F16 path is half the speed of F32 because every element goes through a scalar half-to-float conversion. That is the expected shape of a reference implementation and the first target for SIMD work.

## Design notes

### Kernels

Every operation the forward pass needs lives behind one table of function pointers: matmul for F32, F16 and BF16 weights (single vector and multi-row), RMSNorm, RoPE (adjacent-pair and NeoX layouts, with the Llama 3.x per-dimension frequency factors), softmax, SwiGLU, embedding lookup and causal attention with grouped-query heads, over a whole sequence or over a cache. The scalar implementations under `src/kernels/ref/` are the reference: plain loops, sequential float accumulation, written to be read. Optimized tables slot in without touching any call site and are tested against the scalar one.

### Forward pass

`LlamaModel` reads its configuration from the GGUF metadata, binds every weight with an exact shape and dtype check, and runs the block in the order the paper gives it: RMSNorm, Q/K/V projections, RoPE, causal grouped-query attention, output projection, RMSNorm, SwiGLU feed-forward, each with a residual add. Weights stay memory-mapped. Activations come from one arena sized at load, and a test asserts that the computed size equals the measured high-water mark and that a forward pass performs no heap allocation.

`--dump-layer N` writes one layer's activations as `.axt` tensors, `tests/reference/gen_model_fixtures.py --dump-hf N` writes the same tensors from Hugging Face forward hooks, and `scripts/compare_dumps.py` prints the difference per tensor, for the day a number disagrees.

### KV cache

`KVCache` holds keys and values for every layer, preallocated for the cache length and filled in position order. The storage is an anonymous mapping, so context that is never reached costs no memory. `forward_prefill` appends a chunk of tokens and `forward_one` appends a single token. Both run the same layer code as the cache-free pass, writing new keys and values straight into the cache rows and attending over everything cached so far. Context overflow is an error, never a silent grow.

The defining property is bit-identity: cached decoding produces exactly the floats the cache-free forward produces, at every position, because no reduction is ever reordered on the scalar path.

### Sampling and chat

The sampler runs the stages in llama.cpp's order: repetition penalty over a window of recent tokens, temperature, top-k, top-p, min-p, then one draw. Temperature zero is greedy and skips the rest. The random generator is written in the repo rather than taken from the standard library, because the standard distributions are not bit-specified across implementations and a seeded run should reproduce the same tokens on every platform. Sampling allocates nothing after construction.

Chat prompts come from the model's own template, read from the GGUF metadata and rendered by a small Jinja interpreter. It covers what the SmolLM2, TinyLlama, Llama 3 and Qwen2 templates use, which is loops, conditionals, `set`, attribute and index access, slicing, `in`, the tests `defined` and `none`, the filters `trim` and `length`, and Jinja's whitespace control with the same environment defaults Hugging Face uses. Anything outside the subset falls back to a hard-coded format chosen from the vocabulary's special tokens, with a log line saying why.

The engine keeps the conversation's tokens alongside the cache. When a new prompt arrives it finds the longest common prefix with what the cache holds, truncates to it, and prefills only the rest. A chat template that re-renders the whole transcript every turn therefore costs one turn of prefill.

## Development

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset asan-ubsan && cmake --build --preset asan-ubsan && ctest --preset asan-ubsan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
./build/release/bench/bench_kernels
./build/release/bench/bench_model
scripts/format.sh          # clang-format all sources
cmake --preset tidy && cmake --build --preset tidy
```

`scripts/check_all.sh` runs every gate in one go: format check, build and tests for the debug, release, ASan+UBSan and TSan presets, clang-tidy, the integration tests when models are present, and a benchmark smoke run. It is the required check before a change is merged.

Presets: `debug`, `release`, `release-native` (adds `-march=native`), `asan-ubsan`, `tsan`, `fuzz`, `coverage`, `tidy`.

Dependencies are fetched by CMake at configure time: fmt, CLI11, Catch2 (tests), Google Benchmark (bench). Nothing else.

Tests that need real model files are opt-in: run `scripts/download_models.sh`, then `AXON_MODELS_DIR=$PWD/models ctest --preset release -L integration` (an absolute path, because ctest runs from the build tree). Without the variable they are skipped. The SmolLM2 integration tests take about four minutes on the scalar path.

Fixtures under `tests/reference/fixtures/` are generated by the Python scripts next to them and are small enough to commit:

- GGUF fixtures come from the official `gguf` writer, including a random two-layer LLaMA model in F32, Q8_0 and Q4_0.
- Tokenizer fixtures are tokenizer-only GGUF files plus the reference ids from Hugging Face.
- Kernel fixtures are `.axt` tensors (a 4-byte magic, dims, dtype, raw data) written by NumPy in float64.
- Model fixtures are a NumPy forward over the synthetic model, and for SmolLM2 the Hugging Face argmax per position, last-position logits and greedy continuations. The full per-position logits stay in the gitignored `tests/reference/local/` and are checked when `AXON_LOCAL_FIXTURES` points there.
- Chat template fixtures are the Hugging Face `apply_chat_template` output for three conversations per family, produced from the tokenizers alone.

To add a kernel: declare it in `src/kernels/kernels.h` and `src/kernels/ref/ref.h`, implement the scalar version under `src/kernels/ref/`, register it in the table in `src/kernels/kernels.cpp`, add its reference to the generator and regenerate, then write a fixture test plus at least one property test under `tests/unit/`.

## License

MIT
