# axon-infer

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![C++20](https://img.shields.io/badge/C%2B%2B-20-informational) ![Platform](https://img.shields.io/badge/platform-macOS%20arm64%20%7C%20NEON-lightgrey)

LLM inference engine in C++20, built from scratch. It loads GGUF models from the LLaMA family and runs them on CPU with memory-mapped weights, quantized matmuls, a KV cache, a chat template renderer and a seeded sampler. No ggml, BLAS, OpenMP or other ML runtime. Every component is checked against a reference implementation, and every optimization is benchmarked so you can see what it buys.

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

**Current limitations.** Optimized kernels exist for arm64 with NEON only, so x86 runs the scalar reference table. Decode on small models is bound by per-launch overhead rather than bandwidth, which the fused kernels address next. K-quant formats other than Q6_K load and can be inspected but do not run. There is no HTTP server yet.

## Features

- **GGUF loader** over a read-only memory mapping, validating every offset and length before any accessor can be called, fuzzed with a deterministic corruption replay and a libFuzzer target.
- **Tokenizers** built entirely from GGUF metadata: byte-level BPE with the GPT-2, Llama 3 and Qwen2 pre-tokenizer patterns, and SentencePiece BPE with byte fallback, plus NFC normalization and a streaming decoder that never emits a partial UTF-8 sequence.
- **Forward pass** on a table of kernels with a scalar reference for each, activations from one arena sized at load, and no heap allocation per token. Threads split matmul columns and attention heads so that logits are bit-identical at any thread count.
- **NEON kernels** selected at load from the CPU's features, with `--force-scalar` to run the reference table instead. The integer-dot matmuls and activation quantization are bit-identical to the scalar kernels, so every quantized number is reproducible on either table. The float matmuls are held to a locked tolerance of 8e-7.
- **Quantized weights** in Q8_0, Q4_0, Q4_1 and Q6_K, dequantized inside the matmul, with activations quantized to Q8_0 blocks for an integer dot product or kept in float for comparison.
- **KV cache** with prefill, single-token decode and truncation, bit-identical to the cache-free forward.
- **Sampler** with temperature, top-k, top-p, min-p and a repetition penalty, on a seeded generator that reproduces across platforms.
- **Chat** through the model's own Jinja template, rendered by a small interpreter with per-family fallbacks, reusing the cached prefix of the conversation across turns.
- **Tools**: `axon` (chat, generate, tokenize, bench), `axon-dump` for inspecting any GGUF file, `axon-perplexity` for scoring a model over a text, benchmarks for kernels and the forward pass.

## Supported models and formats

Architectures read from GGUF metadata:

| Architecture | Config | Tokenizer | Forward pass | Verified against Hugging Face |
|---|---|---|---|---|
| `llama` (Llama 2, Llama 3.x, SmolLM2, TinyLlama) | ✅ | ✅ | ✅ | SmolLM2-135M: logits, greedy output, perplexity, chat template. TinyLlama and Llama 3.2 1B: chat |
| `qwen2` (Qwen2.5) | ✅ | ✅ | ✅ (Q/K/V bias, NeoX rotary layout) | tokenizer, chat template, chat |

Weight formats:

| Format | Load and inspect | Run | Activation path |
|---|---|---|---|
| F32, F16, BF16 | ✅ | ✅ | float |
| Q8_0 | ✅ | ✅ | Q8_0 integer dot, or float with `--exact-activations` |
| Q4_0, Q4_1 | ✅ | ✅ | Q8_0 integer dot, or float with `--exact-activations` |
| Q6_K | ✅ | ✅ | float (used by llama.cpp for output and embedding matrices in Q4_0 files) |
| Other K-quants and IQ formats | ✅ (dump only) | no | |

Models exercised by the test suite: SmolLM2-135M-Instruct (F16, Q8_0, Q4_0), TinyLlama-1.1B-Chat (Q4_0), Llama-3.2-1B-Instruct (Q4_0) and Qwen2.5-0.5B-Instruct (Q4_0). All four chat. The "Q4_0" files published for these models also contain Q4_1, Q8_0 and Q6_K tensors, which is why those formats are supported.

Platforms: developed and benchmarked on macOS arm64 (Apple M1), where the NEON table is used when the CPU reports the dot product extension. The rest is portable C++20 that falls back to the scalar table, and Linux x86-64 builds are untested.

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

`/clear` starts a new conversation, `/stats` repeats the last turn's numbers, and `/exit` quits. Sampling is controlled with `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty` and `--seed`. `--system` sets a system prompt, `--ctx` the cache length and `--threads` the thread count, which defaults to one per physical core. `--force-scalar` runs the scalar reference kernels for comparison. Small models decode faster with `--threads 4` on the M1, as the scaling tables below show.

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

```
$ ./build/release/apps/axon/axon chat -m models/Llama-3.2-1B-Instruct-Q4_0.gguf --temp 0
> What is the capital of France? Answer in one sentence.
The capital of France is Paris.
> And Italy?
The capital of Italy is Rome.
```

### Perplexity

`axon-perplexity` scores a model over a text the way llama.cpp's perplexity tool does: the text is cut into chunks of 512 tokens, each chunk is one prefill, and the second half of every chunk is scored with the first half as context. The default text is a committed head of the WikiText-2 raw test split, so the tool works from a clean clone:

```
$ ./build/release/apps/axon-perplexity/axon-perplexity -m models/SmolLM2-135M-Instruct-Q8_0.gguf --chunks 4
model: Smollm2 135M 8k Lc100K Mix1 Ep2 (llama layers 30 embd 576 heads 9 kv 3 ff 1536 vocab 49152 ctx 8192 rope normal 100000 tied)
text: 240000 bytes, 57379 tokens, chunks of 512
[1] 10.3831
[2] 17.1137
[3] 18.3842
[4] 17.7097
perplexity 17.7097 over 1024 scored tokens in 4 chunks, 35.5 s
```

`--chunks` limits the run, `--exact-activations` keeps activations in float for quantized weights, and `--json` prints one object for scripts.

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
| Quantization | the `gguf` package's quantizer, which is bit-exact with ggml | dequantization of every supported block format and activation quantization match bit for bit, and the integer-dot kernels match an integer reference |
| Perplexity | Hugging Face on the same GGUF file, dequantized to float | matches to 0.1 percent on F16 and, with Q8_0 activations, to 0.5 percent on Q8_0 and Q4_0 |
| Chat templates | Hugging Face `apply_chat_template` | twelve renders across four model families match byte for byte |
| Sampler | closed-form expectations | top-k and top-p bounds, temperature zero equals argmax, seeded reproducibility, and draw frequencies within two percent |

Every preset runs the unit tests: debug, release, AddressSanitizer with UndefinedBehaviorSanitizer, and ThreadSanitizer. clang-tidy runs with warnings as errors.

## Benchmarks

Every number comes from `axon bench` through `bench/bench_e2e.sh`, which writes a JSON file per run under `bench/results/` with the host, compiler, commit, model SHA256, kernel table and load average, and the tables here are the output of `bench/render_tables.py` over those files. The method: a fixed 512-token prompt cut from the committed WikiText-2 text, 128 generated tokens, one warm-up, the median of five runs. All runs are on an Apple M1 (four performance and four efficiency cores, 68.25 GB/s peak memory bandwidth).

```
$ ./build/release/apps/axon/axon bench -m models/SmolLM2-135M-Instruct-Q8_0.gguf --threads 4
```

The steps are the optimizations in the order they landed: `06a-static` split work statically across threads on the scalar kernels, `06a-threads` claims it in chunks, and `06b-neon` runs the NEON kernel table. Weight GB/s is the weight bytes one forward pass reads times the decode rate, the number to hold against the machine's bandwidth.

| Step | Model | Threads | Prefill tok/s | Decode tok/s | Weight GB/s | Of peak |
|---|---|---|---|---|---|---|
| 06a-static | Llama-3.2-1B-Instruct-Q4_0 | 1 | 1.9 | 1.90 | 1.45 | 2% |
| 06a-static | Llama-3.2-1B-Instruct-Q4_0 | 8 | 8.6 | 7.69 | 5.88 | 9% |
| 06a-static | SmolLM2-135M-Instruct-Q8_0 | 1 | 55.4 | 44.83 | 6.41 | 9% |
| 06a-static | SmolLM2-135M-Instruct-Q8_0 | 2 | 104.8 | 68.55 | 9.80 | 14% |
| 06a-static | SmolLM2-135M-Instruct-Q8_0 | 4 | 188.6 | 76.78 | 10.98 | 16% |
| 06a-static | SmolLM2-135M-Instruct-Q8_0 | 8 | 209.9 | 45.26 | 6.47 | 9% |
| 06a-static | SmolLM2-135M-Instruct-f16 | 1 | 4.1 | 4.08 | 1.10 | 2% |
| 06a-static | SmolLM2-135M-Instruct-f16 | 2 | 8.1 | 7.59 | 2.04 | 3% |
| 06a-static | SmolLM2-135M-Instruct-f16 | 4 | 13.6 | 12.78 | 3.44 | 5% |
| 06a-static | SmolLM2-135M-Instruct-f16 | 8 | 16.5 | 13.70 | 3.69 | 5% |
| 06a-threads | Llama-3.2-1B-Instruct-Q4_0 | 1 | 1.9 | 1.89 | 1.45 | 2% |
| 06a-threads | Llama-3.2-1B-Instruct-Q4_0 | 4 | 6.4 | 6.17 | 4.72 | 7% |
| 06a-threads | Llama-3.2-1B-Instruct-Q4_0 | 8 | 8.6 | 7.90 | 6.04 | 9% |
| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 1 | 55.4 | 45.07 | 6.45 | 9% |
| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 2 | 104.1 | 72.00 | 10.30 | 15% |
| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 4 | 188.9 | 88.47 | 12.65 | 19% |
| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 8 | 238.9 | 59.05 | 8.45 | 12% |
| 06a-threads | SmolLM2-135M-Instruct-f16 | 1 | 4.1 | 4.08 | 1.10 | 2% |
| 06a-threads | SmolLM2-135M-Instruct-f16 | 2 | 8.1 | 7.64 | 2.06 | 3% |
| 06a-threads | SmolLM2-135M-Instruct-f16 | 4 | 13.6 | 12.73 | 3.42 | 5% |
| 06a-threads | SmolLM2-135M-Instruct-f16 | 8 | 17.2 | 14.80 | 3.98 | 6% |
| 06b-neon | Llama-3.2-1B-Instruct-Q4_0 | 1 | 13.8 | 12.39 | 9.48 | 14% |
| 06b-neon | Llama-3.2-1B-Instruct-Q4_0 | 4 | 41.7 | 33.41 | 25.56 | 37% |
| 06b-neon | Llama-3.2-1B-Instruct-Q4_0 | 8 | 51.0 | 33.20 | 25.40 | 37% |
| 06b-neon | SmolLM2-135M-Instruct-Q8_0 | 1 | 184.1 | 96.56 | 13.81 | 20% |
| 06b-neon | SmolLM2-135M-Instruct-Q8_0 | 2 | 328.4 | 114.73 | 16.41 | 24% |
| 06b-neon | SmolLM2-135M-Instruct-Q8_0 | 4 | 526.2 | 113.36 | 16.21 | 24% |
| 06b-neon | SmolLM2-135M-Instruct-Q8_0 | 8 | 634.0 | 59.25 | 8.47 | 12% |
| 06b-neon | SmolLM2-135M-Instruct-f16 | 1 | 55.2 | 44.32 | 11.93 | 17% |
| 06b-neon | SmolLM2-135M-Instruct-f16 | 2 | 105.3 | 72.09 | 19.40 | 28% |
| 06b-neon | SmolLM2-135M-Instruct-f16 | 4 | 171.5 | 92.01 | 24.76 | 36% |
| 06b-neon | SmolLM2-135M-Instruct-f16 | 8 | 231.1 | 69.57 | 18.72 | 27% |

Scaling of Llama-3.2-1B-Instruct-Q4_0 (Q4_0), step 06b-neon, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 13.8 | 12.39 | 1.00x | 1.00x |
| 4 | 41.7 | 33.41 | 3.01x | 2.70x |
| 8 | 51.0 | 33.20 | 3.69x | 2.68x |

Scaling of SmolLM2-135M-Instruct-Q8_0 (Q8_0), step 06b-neon, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 184.1 | 96.56 | 1.00x | 1.00x |
| 2 | 328.4 | 114.73 | 1.78x | 1.19x |
| 4 | 526.2 | 113.36 | 2.86x | 1.17x |
| 8 | 634.0 | 59.25 | 3.44x | 0.61x |

Scaling of SmolLM2-135M-Instruct-f16 (F16), step 06b-neon, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 55.2 | 44.32 | 1.00x | 1.00x |
| 2 | 105.3 | 72.09 | 1.91x | 1.63x |
| 4 | 171.5 | 92.01 | 3.11x | 2.08x |
| 8 | 231.1 | 69.57 | 4.19x | 1.57x |

What the tables show. On Llama 3.2 1B the NEON table took decode from 1.9 to 12.4 tokens per second on one thread and to 33 on eight, reaching 37 percent of the machine's bandwidth. Most of that step came from one kernel: the file stores its tied output matrix as Q6_K, which alone cost 392 of the 433 ms per token on the scalar path. On SmolLM2 the picture is different: a 135M model's matmuls are microseconds of work, so decode stops scaling past two threads and the per-launch overhead sets the ceiling. That is the next lever. Prefill, with many rows per launch, scales to eight threads on every model.

Kernel rows, one thread, scalar against NEON on the same shapes:

| Kernel | Shape (out x in) | Scalar GFLOP/s | NEON GFLOP/s | Scalar GB/s | NEON GB/s |
|---|---|---|---|---|---|
| F32 matmul | 2048 x 2048 | 2.3 | 24.3 | 4.5 | 48.6 |
| F16 matmul | 2048 x 2048 | 1.1 | 13.9 | 1.1 | 13.9 |
| Q8_0 matmul, Q8_0 activations | 2048 x 2048 | 18.7 | 81.3 | 10.0 | 43.2 |
| Q4_0 matmul, Q8_0 activations | 2048 x 2048 | 15.1 | 63.1 | 4.3 | 17.8 |
| Q6_K matmul, float activations | 2048 x 2048 | 1.4 | 14.7 | 0.6 | 6.0 |

The scalar integer rows are already auto-vectorized by the compiler, which is why the NEON gain there is four times rather than ten. The integer kernels keep the scalar kernel's per-block accumulation order and are bit-identical to it, which is a locked test. The float kernels differ by reduction order within 8e-7, measured first and locked at four times the measurement.

### Quantization accuracy

Measured on SmolLM2-135M-Instruct against its F16 file, over the ten prompt fixtures and the first four WikiText-2 chunks. The Hugging Face column runs the same dequantized weights in float, so the gap between it and the engine is the cost of quantizing activations to Q8_0 blocks, and the gap between rows is the cost of the weight format itself.

| File | Argmax agreement with F16 | Perplexity (engine) | Perplexity (Hugging Face, dequantized) | Delta from F16 |
|---|---|---|---|---|
| F16 | | 17.652 | 17.652 | |
| Q8_0 | 95.2% | 17.710 | 17.665 | +0.06 |
| Q4_0 | 77.9% | 20.185 | 20.194 | +2.53 |

A 135M model loses a lot to Q4_0. That is a property of the model and the format, reproduced by Hugging Face on the same weights, not of the engine. Larger models lose far less, and the perplexity tool makes the trade-off measurable for any file.

### Provenance

| File | Host | CPU | Commit | Date | Model SHA256 |
|---|---|---|---|---|---|
| m1-06a-static-Llama-3.2-1B-Instruct-Q4_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | fa0390e7c043 |
| m1-06a-static-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | fa0390e7c043 |
| m1-06a-static-SmolLM2-135M-Instruct-Q8_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | 5a1395716f79 |
| m1-06a-static-SmolLM2-135M-Instruct-Q8_0-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | 5a1395716f79 |
| m1-06a-static-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | 5a1395716f79 |
| m1-06a-static-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | 5a1395716f79 |
| m1-06a-static-SmolLM2-135M-Instruct-f16-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | f535f83ec568 |
| m1-06a-static-SmolLM2-135M-Instruct-f16-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | f535f83ec568 |
| m1-06a-static-SmolLM2-135M-Instruct-f16-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | f535f83ec568 |
| m1-06a-static-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 34af445d52af | 2026-10-03 | f535f83ec568 |
| m1-06a-threads-Llama-3.2-1B-Instruct-Q4_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | fa0390e7c043 |
| m1-06a-threads-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | fa0390e7c043 |
| m1-06a-threads-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | fa0390e7c043 |
| m1-06a-threads-SmolLM2-135M-Instruct-Q8_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | 5a1395716f79 |
| m1-06a-threads-SmolLM2-135M-Instruct-Q8_0-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | 5a1395716f79 |
| m1-06a-threads-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | 5a1395716f79 |
| m1-06a-threads-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | 5a1395716f79 |
| m1-06a-threads-SmolLM2-135M-Instruct-f16-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | f535f83ec568 |
| m1-06a-threads-SmolLM2-135M-Instruct-f16-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | f535f83ec568 |
| m1-06a-threads-SmolLM2-135M-Instruct-f16-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | f535f83ec568 |
| m1-06a-threads-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 7569e589a155 | 2026-10-04 | f535f83ec568 |
| m1-06b-neon-Llama-3.2-1B-Instruct-Q4_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | d000fa7ba6b7 (dirty) | 2026-10-10 | fa0390e7c043 |
| m1-06b-neon-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | d000fa7ba6b7 (dirty) | 2026-10-10 | fa0390e7c043 |
| m1-06b-neon-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | d000fa7ba6b7 (dirty) | 2026-10-10 | fa0390e7c043 |
| m1-06b-neon-SmolLM2-135M-Instruct-Q8_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | 5a1395716f79 |
| m1-06b-neon-SmolLM2-135M-Instruct-Q8_0-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | 5a1395716f79 |
| m1-06b-neon-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | 5a1395716f79 |
| m1-06b-neon-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | 5a1395716f79 |
| m1-06b-neon-SmolLM2-135M-Instruct-f16-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | f535f83ec568 |
| m1-06b-neon-SmolLM2-135M-Instruct-f16-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | f535f83ec568 |
| m1-06b-neon-SmolLM2-135M-Instruct-f16-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | f535f83ec568 |
| m1-06b-neon-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4c3562c65cc1 | 2026-10-06 | f535f83ec568 |

## Design notes

### Kernels

Two tables implement the same interface. The scalar table is the reference, plain loops written to be read, and the NEON table overrides the matmuls and activation quantization with intrinsics compiled only into those files, with the dot product and half-precision extensions enabled there alone. The table is chosen at load from the CPU's reported features and can be forced to scalar on any command. RMSNorm, SiLU and softmax stay scalar on purpose: they are under one part in a thousand of a token's arithmetic, and keeping them scalar is what keeps the quantized models bit-identical across tables.

Every operation the forward pass needs lives behind one table of function pointers: matmul for F32, F16 and BF16 weights (single vector and multi-row), RMSNorm, RoPE (adjacent-pair and NeoX layouts, with the Llama 3.x per-dimension frequency factors), softmax, SwiGLU, embedding lookup and causal attention with grouped-query heads, over a whole sequence or over a cache. The scalar implementations under `src/kernels/ref/` are the reference: plain loops, sequential float accumulation, written to be read. Optimized tables slot in without touching any call site and are tested against the scalar one.

### Forward pass

Threads split the output columns of every matmul, the heads of the attention and the rows of activation quantization, claiming chunks dynamically so the M1's slower efficiency cores do not pace each launch. No partial sum ever crosses a thread, so the logits are bit-identical at any thread count, and that equality is a test on every weight format rather than a tolerance. There is no deterministic mode because nothing needs one yet.

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

Tests that need real model files are opt-in: run `scripts/download_models.sh`, then `AXON_MODELS_DIR=$PWD/models ctest --preset release -L integration` (an absolute path, because ctest runs from the build tree). Without the variable they are skipped. The integration tests take under four minutes with the default thread count, most of it the F16 perplexity pass.

Fixtures under `tests/reference/fixtures/` are generated by the Python scripts next to them and are small enough to commit:

- GGUF fixtures come from the official `gguf` writer, including a random two-layer LLaMA model in F32, Q8_0 and Q4_0.
- Tokenizer fixtures are tokenizer-only GGUF files plus the reference ids from Hugging Face.
- Kernel fixtures are `.axt` tensors (a 4-byte magic, dims, dtype, raw data) written by NumPy in float64.
- Model fixtures are a NumPy forward over the synthetic model, and for SmolLM2 the Hugging Face argmax per position, last-position logits and greedy continuations. The full per-position logits stay in the gitignored `tests/reference/local/` and are checked when `AXON_LOCAL_FIXTURES` points there.
- Chat template fixtures are the Hugging Face `apply_chat_template` output for three conversations per family, produced from the tokenizers alone.
- Benchmark results under `bench/results/` are written by `bench/bench_e2e.sh` and never edited. `bench/render_tables.py` renders them, and `--self-test` checks the renderer in the check script.
- Quantization fixtures are rows and small matrices quantized by the `gguf` package, with its dequantized values alongside.
- The perplexity text is the head of the WikiText-2 raw test split (Merity et al., "Pointer Sentinel Mixture Models", 2016, CC BY-SA 3.0), written by `scripts/make_perplexity_fixture.py`, with Hugging Face reference numbers for the SmolLM2 files from `tests/reference/gen_perplexity_reference.py`.

To add a kernel: declare it in `src/kernels/kernels.h` and `src/kernels/ref/ref.h`, implement the scalar version under `src/kernels/ref/`, register it in the table in `src/kernels/kernels.cpp`, add its reference to the generator and regenerate, then write a fixture test plus at least one property test under `tests/unit/`.

## License

MIT
