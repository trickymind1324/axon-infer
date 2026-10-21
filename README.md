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

**Current limitations.** Optimized kernels exist for arm64 with NEON only, so x86 runs the scalar reference table. Decode on a 135M model is still paced by the cost of handing work to threads rather than by bandwidth, so small models decode fastest at four threads and lose speed at eight. K-quant formats other than Q6_K load and can be inspected but do not run. The HTTP server decodes up to four requests together and queues the rest, with speculative decoding and context shift still to come.

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

`/clear` starts a new conversation, `/stats` repeats the last turn's numbers, and `/exit` quits. Sampling is controlled with `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty` and `--seed`. `--system` sets a system prompt, `--ctx` the cache length and `--threads` the thread count, which defaults to one per physical core. `--force-scalar` runs the scalar reference kernels for comparison. `--cache f16` stores keys and values in half precision, which halves the cache memory and the cache bytes read per token at a measured, locked cost in accuracy. Small models decode faster with `--threads 4` on the M1, as the scaling tables below show.

### Serve

`axon-server` puts one model behind the OpenAI chat completions API, so any client that takes a base URL can talk to it:

```
$ ./build/release/apps/axon-server/axon-server -m models/SmolLM2-135M-Instruct-f16.gguf --port 8080
model: Smollm2 135M 8k Lc100K Mix1 Ep2 (llama layers 30 embd 576 heads 9 kv 3 ff 1536 vocab 49152 ctx 8192 rope normal 100000 tied), loaded in 0.02 s, 8 threads, neon kernels, context 4096, f32 cache
serving 'SmolLM2-135M-Instruct-f16' at http://127.0.0.1:8080/v1 (no authentication), one request at a time, queue of 16
```

```
$ curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32, "temperature": 0
  }'
{"choices":[{"finish_reason":"length","index":0,"message":{"content":"The capital of France is Paris. It's a city that has been a beacon for art, culture, and innovation since ancient times. The French people have always","role":"assistant"}}],"created":1792511424,"id":"chatcmpl-MYIiDtVP49yGqbxuFuKzNxSz","model":"SmolLM2-135M-Instruct-f16","object":"chat.completion","usage":{"completion_tokens":32,"prompt_tokens":37,"total_tokens":69}}
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="SmolLM2-135M-Instruct-f16",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The routes are `POST /v1/chat/completions` (streaming over server-sent events when `stream` is true), `POST /v1/completions` for a raw prompt, `GET /v1/models` and `GET /health`, which reports the queue depth and the requests served. Requests take the OpenAI fields `max_tokens`, `temperature`, `top_p`, `seed`, `stop` (up to four strings, matched on the text), `stream` and `stream_options.include_usage`, plus the sampler's `top_k`, `min_p` and `repetition_penalty` as extensions. Fields the server cannot honour, such as `n` above one, `tools` or `logprobs`, are refused with 400 and the field's name rather than ignored. Errors come in the OpenAI error shape.

Up to `--max-batch` requests (default four) decode together: each has its own cache of `--ctx` positions, and one forward pass per step advances all of them, reading the weights once for the batch. Requests beyond the slots wait in arrival order and `--max-queue` bounds how many may wait before new ones get 503 with a `Retry-After` header. A new prompt is prefilled one chunk per step while the others keep decoding, so running streams pause for a chunk rather than a whole prompt. A chat client that resends the whole conversation each turn gets back the slot that still holds its prefix and pays one turn of prefill. A client that disconnects mid-stream stops its generation within one token. `/health` reports the slots, the sequences running and the steps that batched, and every completion carries the batch size it ran in. Memory is slots times context: four Llama 3.2 1B slots at 4096 positions can touch 1 GB of F32 cache, half with `--cache f16`, and the reservation is lazy so untouched context costs nothing. `--api-key` requires `Authorization: Bearer <key>` on the API routes, `--ctx` sets the context the prompt and completion must fit, and `--threads`, `--cache` and `--force-scalar` work as in `axon`.

The Dockerfile builds the server on Debian with clang and runs it in an image that holds only the two binaries and the C++ runtime:

```
$ docker build -t axon-infer .
$ docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" axon-infer -m /models/SmolLM2-135M-Instruct-f16.gguf
```

The image is 144 MB. On an arm64 host it gets the NEON kernel table, on x86 the scalar one. On macOS, Docker Desktop must be allowed to share the folder that holds the models, and the Desktop folder is not shared by default.

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

The steps are the optimizations in the order they landed: `06a-static` split work statically across threads on the scalar kernels, `06a-threads` claims it in chunks, `06b-neon` runs the NEON kernel table, `06d-fused` merges the launches that share an input and keeps idle workers spinning for a short budget, `06d-f16cache` is the same build with `--cache f16`, `06e-gemm` runs prefill through multi-row kernel tiles and a NEON attention kernel, and `08b-batch` decodes several streams in one pass. Weight GB/s is the weight bytes one forward pass reads times the decode rate, the number to hold against the machine's bandwidth.

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
| 06d-f16cache | Llama-3.2-1B-Instruct-Q4_0 | 8 | 52.9 | 38.12 | 29.17 | 43% |
| 06d-f16cache | SmolLM2-135M-Instruct-Q8_0 | 8 | 737.0 | 116.27 | 16.63 | 24% |
| 06d-f16cache | SmolLM2-135M-Instruct-f16 | 8 | 266.2 | 73.19 | 19.70 | 29% |
| 06d-fused | Llama-3.2-1B-Instruct-Q4_0 | 1 | 13.5 | 12.06 | 9.23 | 14% |
| 06d-fused | Llama-3.2-1B-Instruct-Q4_0 | 4 | 38.8 | 32.85 | 25.14 | 37% |
| 06d-fused | Llama-3.2-1B-Instruct-Q4_0 | 8 | 47.9 | 33.38 | 25.55 | 37% |
| 06d-fused | SmolLM2-135M-Instruct-Q8_0 | 1 | 177.4 | 92.61 | 13.25 | 19% |
| 06d-fused | SmolLM2-135M-Instruct-Q8_0 | 2 | 328.0 | 129.09 | 18.46 | 27% |
| 06d-fused | SmolLM2-135M-Instruct-Q8_0 | 4 | 548.2 | 155.94 | 22.30 | 33% |
| 06d-fused | SmolLM2-135M-Instruct-Q8_0 | 8 | 628.4 | 98.49 | 14.09 | 21% |
| 06d-fused | SmolLM2-135M-Instruct-f16 | 1 | 53.9 | 43.29 | 11.65 | 17% |
| 06d-fused | SmolLM2-135M-Instruct-f16 | 2 | 103.9 | 74.39 | 20.02 | 29% |
| 06d-fused | SmolLM2-135M-Instruct-f16 | 4 | 170.3 | 107.23 | 28.86 | 42% |
| 06d-fused | SmolLM2-135M-Instruct-f16 | 8 | 220.3 | 68.78 | 18.51 | 27% |
| 06e-gemm | Llama-3.2-1B-Instruct-Q4_0 | 1 | 34.5 | 13.27 | 10.15 | 15% |
| 06e-gemm | Llama-3.2-1B-Instruct-Q4_0 | 4 | 97.7 | 37.47 | 28.68 | 42% |
| 06e-gemm | Llama-3.2-1B-Instruct-Q4_0 | 8 | 117.8 | 40.73 | 31.17 | 46% |
| 06e-gemm | SmolLM2-135M-Instruct-Q8_0 | 1 | 383.4 | 141.13 | 20.18 | 30% |
| 06e-gemm | SmolLM2-135M-Instruct-Q8_0 | 2 | 662.5 | 178.72 | 25.56 | 37% |
| 06e-gemm | SmolLM2-135M-Instruct-Q8_0 | 4 | 1138.4 | 208.78 | 29.86 | 44% |
| 06e-gemm | SmolLM2-135M-Instruct-Q8_0 | 8 | 1318.1 | 132.94 | 19.01 | 28% |
| 06e-gemm | SmolLM2-135M-Instruct-f16 | 1 | 165.1 | 50.81 | 13.67 | 20% |
| 06e-gemm | SmolLM2-135M-Instruct-f16 | 2 | 300.5 | 91.69 | 24.67 | 36% |
| 06e-gemm | SmolLM2-135M-Instruct-f16 | 4 | 482.2 | 136.16 | 36.64 | 54% |
| 06e-gemm | SmolLM2-135M-Instruct-f16 | 8 | 644.3 | 95.85 | 25.79 | 38% |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 4 | 85.0 | 32.73 | 25.04 | 37% |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 8 | 105.1 | 30.87 | 23.62 | 35% |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 4 | 1125.6 | 193.84 | 27.72 | 41% |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 8 | 1091.9 | 100.88 | 14.43 | 21% |

Scaling of Llama-3.2-1B-Instruct-Q4_0 (Q4_0), step 06e-gemm, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 34.5 | 13.27 | 1.00x | 1.00x |
| 4 | 97.7 | 37.47 | 2.83x | 2.82x |
| 8 | 117.8 | 40.73 | 3.41x | 3.07x |

Scaling of SmolLM2-135M-Instruct-Q8_0 (Q8_0), step 06e-gemm, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 383.4 | 141.13 | 1.00x | 1.00x |
| 2 | 662.5 | 178.72 | 1.73x | 1.27x |
| 4 | 1138.4 | 208.78 | 2.97x | 1.48x |
| 8 | 1318.1 | 132.94 | 3.44x | 0.94x |

Scaling of SmolLM2-135M-Instruct-f16 (F16), step 06e-gemm, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 165.1 | 50.81 | 1.00x | 1.00x |
| 2 | 300.5 | 91.69 | 1.82x | 1.80x |
| 4 | 482.2 | 136.16 | 2.92x | 2.68x |
| 8 | 644.3 | 95.85 | 3.90x | 1.89x |

What the tables show. On Llama 3.2 1B the NEON table took decode from 1.9 to 12.4 tokens per second on one thread and to 33 on eight, reaching 37 percent of the machine's bandwidth. Most of that step came from one kernel: the file stores its tied output matrix as Q6_K, which alone cost 392 of the 433 ms per token on the scalar path. On SmolLM2 the picture is different: a 135M model's matmuls are microseconds of work, so decode stops scaling past two threads and the per-launch overhead sets the ceiling. Prefill, with many rows per launch, scales to eight threads on every model.

The fused step went after that ceiling from two sides. The q, k and v projections became one launch, gate and up became one launch with the SiLU applied by the thread that produced both halves, and the activation quantization for a single row runs inline, which took a SmolLM2 token from about 271 launches to 151. Idle workers now spin for 200 microseconds before sleeping, set from a kernel benchmark that times a launch after a 30 microsecond single-threaded gap: with the budget a launch costs 0.3 microseconds on two threads and 0.8 on four, against 4 and 7 without it. On SmolLM2 Q8_0 that moved decode from 113 to 156 tokens per second on four threads and from 115 to 129 on two. One thread and Llama 3.2 1B did not move, because neither was launch bound. Eight threads still lose on the small model: the benchmark shows a launch costing 22 microseconds there against 14 asleep, because every core is busy and the efficiency cores answer slowly, so the thread count rather than the budget is the lever and four is the right default on this machine for small models.

The `06d-f16cache` rows are the same build with `--cache f16` at eight threads. Decode moved from 98.5 to 116.3 tokens per second on SmolLM2 Q8_0, from 68.8 to 73.2 on SmolLM2 F16 and from 33.4 to 38.1 on Llama 3.2 1B, with the cache bytes read per token halved. Eight-thread runs on this machine vary more between sittings than the other thread counts, so the direction is the result and the exact gap is not.

A software prefetch of the weight rows ahead of the NEON Q8_0 and Q4_0 kernels was tried at three distances and removed. Every distance slowed the Q8_0 row by 4 to 13 percent, and the one Q4_0 gain was equal to that run's own spread. The hardware prefetcher already follows the four sequential row streams.

The `06e-gemm` step is prefill and attention. Prefill used to run the single-row matmul once per prompt row, so every weight block was unpacked once per row. It now calls multi-row kernels that on the NEON table are register tiles: a weight block is loaded and unpacked once and applied to four or eight activation rows, each row keeping the single-row kernel's accumulators and fused multiply-add order, so the tiles are bit-identical to the old path row by row and every locked identity held without a tolerance change. The second half of the step is a NEON kernel for the F32 attention, which had been the scalar reference in both tables and was the largest slot on SmolLM2 in both passes, and a split of the attention launch into head by row-group items so a prompt spreads over more items than there are heads. Prefill on Llama 3.2 1B went from 13.5 to 34.5 tokens per second on one thread and from 47.9 to 117.8 on eight, on SmolLM2 Q8_0 from 548 to 1138 on four threads, and on SmolLM2 F16 from 53.9 to 165 on one. Decode moved where attention was the cost: SmolLM2 Q8_0 from 156 to 209 tokens per second on four threads, Llama 3.2 1B from 33.4 to 40.7 on eight. The profile of a Llama 3.2 1B prompt token on eight threads now reads 49 percent in the Q6_K output matrix, 21 in gate and up, 15 in down and 6 in attention. The tile heights were chosen by measurement, 4 rows for F32 and Q6_K and 8 for F16, Q8_0 and Q4_0, where more rows spill registers.

Kernel rows, one thread, scalar against NEON on the same shapes:

| Kernel | Shape (out x in) | Scalar GFLOP/s | NEON GFLOP/s | Scalar GB/s | NEON GB/s |
|---|---|---|---|---|---|
| F32 matmul | 2048 x 2048 | 2.3 | 24.3 | 4.5 | 48.6 |
| F16 matmul | 2048 x 2048 | 1.1 | 13.9 | 1.1 | 13.9 |
| Q8_0 matmul, Q8_0 activations | 2048 x 2048 | 18.7 | 81.3 | 10.0 | 43.2 |
| Q4_0 matmul, Q8_0 activations | 2048 x 2048 | 15.1 | 63.1 | 4.3 | 17.8 |
| Q6_K matmul, float activations | 2048 x 2048 | 1.4 | 14.7 | 0.6 | 6.0 |
| Attention, 9 heads over 3 kv heads of 64, 512 positions, one query row | | 7.9 | 18.4 | | |

Multi-row rows, 32 activation rows against the same weights, the shape a prefill chunk presents. The "per row" column is the single-row NEON kernel called once per row, which is what prefill did before the tiles:

| Kernel | Shape (rows x out x in) | Scalar GFLOP/s | NEON per row GFLOP/s | NEON tile GFLOP/s |
|---|---|---|---|---|
| F32 matmul | 32 x 2048 x 2048 | 2.2 | 22 | 44 |
| F16 matmul | 32 x 2048 x 2048 | 1.1 | 14 | 52 |
| Q8_0 matmul, Q8_0 activations | 32 x 2048 x 2048 | 18.5 | 81 | 188 |
| Q4_0 matmul, Q8_0 activations | 32 x 2048 x 2048 | 15.1 | 63 | 170 |
| Q6_K matmul, float activations | 32 x 2048 x 2048 | 1.4 | 15 | 38 |

The scalar integer rows are already auto-vectorized by the compiler, which is why the NEON gain there is four times rather than ten. The integer kernels keep the scalar kernel's per-block accumulation order and are bit-identical to it, which is a locked test. The float kernels differ by reduction order within 8e-7, measured first and locked at four times the measurement.

### Against llama.cpp

`scripts/compare_llamacpp.sh` builds llama.cpp at a pinned commit, CPU only, and runs `llama-bench -p 512 -n 128 -r 5` on the same files, which is the engine's prompt length, generation length and repetition count. The rows go into `bench/results/` with the step `llamacpp` and the table below is rendered from them next to the engine's latest step. Two differences are stated rather than adjusted for: `tg128` generates from an empty context while the engine decodes after the 512-token prompt, so the engine's decode carries the attention over those positions, and llama.cpp repacks Q4_0 weights into its interleaved arm64 layout at load, a different kernel family from a plain Q4_0 dot product. llama.cpp also defaults to an fp16 cache, where these engine rows use the F32 cache.

| Model | Threads | Engine step | Prefill tok/s (engine) | Prefill tok/s (llama.cpp) | Share | Decode tok/s (engine) | Decode tok/s (llama.cpp) | Share |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_0 | 4 | 06e-gemm | 97.7 | 336.5 | 29% | 37.47 | 67.20 | 56% |
| Llama-3.2-1B-Instruct-Q4_0 | 8 | 06e-gemm | 117.8 | 347.6 | 34% | 40.73 | 48.29 | 84% |
| SmolLM2-135M-Instruct-Q8_0 | 4 | 06e-gemm | 1138.4 | 1853.9 | 61% | 208.78 | 317.83 | 66% |
| SmolLM2-135M-Instruct-Q8_0 | 8 | 06e-gemm | 1318.1 | 1721.6 | 77% | 132.94 | 179.27 | 74% |

On Llama 3.2 1B Q4_0 at eight threads the engine decodes at 84 percent of llama.cpp and prefills at 34 percent, under the 40 percent this step set itself. The profile says where: half of a prompt token is the tied Q6_K output matrix, which the engine dequantizes to float and dots at 38 GFLOP/s per core, where llama.cpp quantizes the activations to 8-bit blocks and runs that matrix on integer dot products like the Q4_0 and Q8_0 paths here. The next lever is that integer activation path for Q6_K, a numerics change that gets its own Hugging Face gates. The rest of the gap is the Q4_0 tiles against llama.cpp's repacked layout. On the small model the engine is within a third of llama.cpp on both passes at four threads, and closer at eight because llama.cpp loses more there.

### Batching

`axon bench --batch N` admits N copies of the prompt into the batch engine, prefills them one chunk per step while the earlier ones decode, then decodes them as one batch for 128 tokens. The rates count every stream, and the single-stream rows are the same build at batch one in the same sitting. Decode speedup is aggregate throughput over the single stream.

What the table shows. One weight pass serves every row of a step, so the second stream on Llama 3.2 1B is cheap: batch 2 decodes 1.6 times the tokens of a single stream, batch 4 between 2.0 and 2.2 times and batch 8 between 2.3 and 2.6 times, for 79 tokens per second aggregate on eight threads at 10 per stream. The curve flattens because a step with eight rows is no longer bound by reading the weights: the matmuls run at the rates prefill gets from the multi-row tiles, and the attention runs once per sequence, so aggregate decode converges towards the model's prefill throughput. On SmolLM2 the gain is smaller, 2.1 times at batch 8 on four threads, because attention over the 512-token context is already half of its decoded token and does not batch. Eight threads lose to four on the small model at every batch size, as they do on a single stream. The first run of this set exposed that a batch smaller than a tile's height fell to a one-row tail in the tiled kernels, so two rows cost two weight passes: Llama batch 2 gave 37 tokens per second before that fix and 53 after it.

| Step | Model | Threads | Batch | Prefill tok/s | Decode tok/s | Per stream | Decode speedup |
|---|---|---|---|---|---|---|---|
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 4 | 2 | 80.9 | 52.96 | 26.48 | 1.62x |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 4 | 4 | 77.7 | 66.21 | 16.55 | 2.02x |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 4 | 8 | 76.3 | 75.35 | 9.42 | 2.30x |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 8 | 2 | 101.7 | 49.65 | 24.82 | 1.61x |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 8 | 4 | 96.0 | 66.83 | 16.71 | 2.16x |
| 08b-batch | Llama-3.2-1B-Instruct-Q4_0 | 8 | 8 | 93.7 | 79.13 | 9.89 | 2.56x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 4 | 2 | 1080.0 | 295.12 | 147.56 | 1.52x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 4 | 4 | 985.0 | 369.31 | 92.33 | 1.91x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 4 | 8 | 846.6 | 416.22 | 52.03 | 2.15x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 8 | 2 | 1119.2 | 127.92 | 63.96 | 1.27x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 8 | 4 | 1007.8 | 197.19 | 49.30 | 1.95x |
| 08b-batch | SmolLM2-135M-Instruct-Q8_0 | 8 | 8 | 868.5 | 271.40 | 33.92 | 2.69x |

### Quantization accuracy

Measured on SmolLM2-135M-Instruct against its F16 file, over the ten prompt fixtures and the first four WikiText-2 chunks. The Hugging Face column runs the same dequantized weights in float, so the gap between it and the engine is the cost of quantizing activations to Q8_0 blocks, and the gap between rows is the cost of the weight format itself.

| File | Argmax agreement with F16 | Perplexity (engine) | Perplexity (Hugging Face, dequantized) | Delta from F16 |
|---|---|---|---|---|
| F16 | | 17.652 | 17.652 | |
| Q8_0 | 97.2% | 17.677 | 17.665 | +0.03 |
| Q4_0 | 80.0% | 20.226 | 20.194 | +2.57 |

These are the default NEON table's numbers. The scalar table gives 95.2 percent and 17.710 for Q8_0 and 77.9 percent and 20.185 for Q4_0, and the F16 file gives 17.652 on both. The gap between the tables is the NEON attention's reduction order, a perturbation of about one part in ten million that the Q8_0 activation quantization in front of the next matmul turns into a different rounding of a few activations per layer. Neither table is more right than the other, and both sit within the locked tolerance of the Hugging Face reference.

A 135M model loses a lot to Q4_0. That is a property of the model and the format, reproduced by Hugging Face on the same weights, not of the engine. Larger models lose far less, and the perplexity tool makes the trade-off measurable for any file.

The fp16 cache is a different kind of trade. On the F16 file it leaves every argmax over the 145 fixture positions unchanged and moves the perplexity from 17.6516 to 17.6515, for half the cache memory.

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
| m1-06d-f16cache-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 93012f7c8ab4 (dirty) | 2026-10-11 | fa0390e7c043 |
| m1-06d-f16cache-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 93012f7c8ab4 (dirty) | 2026-10-11 | 5a1395716f79 |
| m1-06d-f16cache-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 93012f7c8ab4 (dirty) | 2026-10-11 | f535f83ec568 |
| m1-06d-fused-Llama-3.2-1B-Instruct-Q4_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | fa0390e7c043 |
| m1-06d-fused-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | fa0390e7c043 |
| m1-06d-fused-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | fa0390e7c043 |
| m1-06d-fused-SmolLM2-135M-Instruct-Q8_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | 5a1395716f79 |
| m1-06d-fused-SmolLM2-135M-Instruct-Q8_0-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | 5a1395716f79 |
| m1-06d-fused-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | 5a1395716f79 |
| m1-06d-fused-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | 5a1395716f79 |
| m1-06d-fused-SmolLM2-135M-Instruct-f16-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | f535f83ec568 |
| m1-06d-fused-SmolLM2-135M-Instruct-f16-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | f535f83ec568 |
| m1-06d-fused-SmolLM2-135M-Instruct-f16-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | f535f83ec568 |
| m1-06d-fused-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | 4a9a4a2252ad | 2026-10-11 | f535f83ec568 |
| m1-06e-gemm-Llama-3.2-1B-Instruct-Q4_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | fa0390e7c043 |
| m1-06e-gemm-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | fa0390e7c043 |
| m1-06e-gemm-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | fa0390e7c043 |
| m1-06e-gemm-SmolLM2-135M-Instruct-Q8_0-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | 5a1395716f79 |
| m1-06e-gemm-SmolLM2-135M-Instruct-Q8_0-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | 5a1395716f79 |
| m1-06e-gemm-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | 5a1395716f79 |
| m1-06e-gemm-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | 5a1395716f79 |
| m1-06e-gemm-SmolLM2-135M-Instruct-f16-t1 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | f535f83ec568 |
| m1-06e-gemm-SmolLM2-135M-Instruct-f16-t2 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | f535f83ec568 |
| m1-06e-gemm-SmolLM2-135M-Instruct-f16-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | f535f83ec568 |
| m1-06e-gemm-SmolLM2-135M-Instruct-f16-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | c3fd65d982ca | 2026-10-20 | f535f83ec568 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | fa0390e7c043 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-08b-batch-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | df6726c76df8 | 2026-10-21 | 5a1395716f79 |
| m1-llamacpp-Llama-3.2-1B-Instruct-Q4_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | llama.cpp 1ba568c5ad5c | 2026-10-20 | fa0390e7c043 |
| m1-llamacpp-Llama-3.2-1B-Instruct-Q4_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | llama.cpp 1ba568c5ad5c | 2026-10-20 | fa0390e7c043 |
| m1-llamacpp-SmolLM2-135M-Instruct-Q8_0-t4 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | llama.cpp 1ba568c5ad5c | 2026-10-20 | 5a1395716f79 |
| m1-llamacpp-SmolLM2-135M-Instruct-Q8_0-t8 | m1 | Apple M1 (arm64) cores=8/8 cacheline=128 features=neon,dotprod,fp16 | llama.cpp 1ba568c5ad5c | 2026-10-20 | 5a1395716f79 |

## Design notes

### Kernels

Two tables implement the same interface. The scalar table is the reference, plain loops written to be read, and the NEON table overrides the matmuls, the activation quantization and the attention with intrinsics compiled only into those files, with the dot product and half-precision extensions enabled there alone. Prefill calls the multi-row entries, which on the NEON table are register tiles: each weight block is loaded, and for the quantized formats unpacked, once per four or eight activation rows, while every row keeps the accumulators and the fused multiply-add order of the single-row kernel, so a row's result does not depend on how many rows share the call. That is what keeps cached decoding bit-identical to the cache-free forward on both tables. The table is chosen at load from the CPU's reported features and can be forced to scalar on any command. RMSNorm, SiLU and softmax stay scalar on purpose: they are under one part in a thousand of a token's arithmetic, and keeping them scalar is what keeps the quantized models bit-identical across tables.

Every operation the forward pass needs lives behind one table of function pointers: matmul for F32, F16 and BF16 weights (single vector and multi-row), RMSNorm, RoPE (adjacent-pair and NeoX layouts, with the Llama 3.x per-dimension frequency factors), softmax, SwiGLU, embedding lookup and causal attention with grouped-query heads, over a whole sequence or over a cache. The scalar implementations under `src/kernels/ref/` are the reference: plain loops, sequential float accumulation, written to be read. Optimized tables slot in without touching any call site and are tested against the scalar one.

### Forward pass

Threads split the output columns of every matmul, the heads of the attention and the rows of activation quantization, claiming chunks dynamically so the M1's slower efficiency cores do not pace each launch. Launches that share an input row are one launch: q, k and v over the union of their columns, gate and up over column pairs with the SiLU product applied in the epilogue, each output element still produced by the same kernel call on the same data. A worker that runs out of work spins for a wall-clock budget before it sleeps, because the gaps between launches are tens of microseconds and a sleeping worker costs a wake-up on every one. No partial sum ever crosses a thread, so the logits are bit-identical at any thread count and whichever launches are fused, and that equality is a test on every weight format rather than a tolerance. There is no deterministic mode because nothing needs one yet.

`LlamaModel` reads its configuration from the GGUF metadata, binds every weight with an exact shape and dtype check, and runs the block in the order the paper gives it: RMSNorm, Q/K/V projections, RoPE, causal grouped-query attention, output projection, RMSNorm, SwiGLU feed-forward, each with a residual add. Weights stay memory-mapped. Activations come from one arena sized at load, and a test asserts that the computed size equals the measured high-water mark and that a forward pass performs no heap allocation.

`--dump-layer N` writes one layer's activations as `.axt` tensors, `tests/reference/gen_model_fixtures.py --dump-hf N` writes the same tensors from Hugging Face forward hooks, and `scripts/compare_dumps.py` prints the difference per tensor, for the day a number disagrees.

### KV cache

`KVCache` holds keys and values for every layer, preallocated for the cache length and filled in position order. The storage is an anonymous mapping, so context that is never reached costs no memory. `forward_prefill` appends a chunk of tokens and `forward_one` appends a single token. Both run the same layer code as the cache-free pass, writing new keys and values straight into the cache rows and attending over everything cached so far. Context overflow is an error, never a silent grow.

The defining property is bit-identity: with the default F32 cache, cached decoding produces exactly the floats the cache-free forward produces, at every position, because no reduction is ever reordered on the scalar path.

The cache can also hold keys and values as fp16. New rows are computed in float, rounded once on the way in, and the attention kernel widens them as it reads, in the same loop order as the F32 kernel. The arena is sized for the cache type the model was loaded with, and a cache of the other type is refused. The cost of the rounding is measured and locked: on the synthetic models the logits move by at most 1.7e-3 of their range, on SmolLM2 by 5.8e-4 with every argmax unchanged over the prompt fixtures, and the perplexity over the WikiText-2 chunks goes from 17.6516 to 17.6515. The NEON fp16 attention agrees with the scalar one within 8e-7 of the logit range, measured first and locked at four times the measurement like the other NEON kernels.

### Sampling and chat

The sampler runs the stages in llama.cpp's order: repetition penalty over a window of recent tokens, temperature, top-k, top-p, min-p, then one draw. Temperature zero is greedy and skips the rest. The random generator is written in the repo rather than taken from the standard library, because the standard distributions are not bit-specified across implementations and a seeded run should reproduce the same tokens on every platform. Sampling allocates nothing after construction.

Chat prompts come from the model's own template, read from the GGUF metadata and rendered by a small Jinja interpreter. It covers what the SmolLM2, TinyLlama, Llama 3 and Qwen2 templates use, which is loops, conditionals, `set`, attribute and index access, slicing, `in`, the tests `defined` and `none`, the filters `trim` and `length`, and Jinja's whitespace control with the same environment defaults Hugging Face uses. Anything outside the subset falls back to a hard-coded format chosen from the vocabulary's special tokens, with a log line saying why.

The engine keeps the conversation's tokens alongside the cache. When a new prompt arrives it finds the longest common prefix with what the cache holds, truncates to it, and prefills only the rest. A chat template that re-renders the whole transcript every turn therefore costs one turn of prefill.

### Server

Several requests decode in one forward pass. Each row of a batched step runs the same instruction sequence the sequence would run alone: the multi-row matmul tiles are tested row by row against the single-row kernels, the attention computes each query row on its own against its own cache, and the norms, RoPE and residual adds are per row. So a batched step is bit-identical to decoding each sequence alone, which the tests assert as equality on every synthetic format, both kernel tables, several thread counts and both cache types, with sequences at unequal positions. One Sequence type holds a generation (its cache, sampler, decoder and end rules) and both the single-sequence engine and the batch engine drive it, so there is one implementation of the loop. The scheduler admits requests into free slots in arrival order, prefills one chunk of the oldest pending prompt per step, then advances everyone in decode together.

The server is a thin layer that knows nothing the engine does not. Request bodies are parsed into the engine's terms by code that is tested on JSON alone, chat messages go through the same template path as `axon chat`, and stop strings are matched on the decoded text by a holdback buffer that releases only what can no longer begin a stop string. One worker thread owns the engine, because the engine holds one cache and one arena and a second generation at once would mean a second copy of both on an 8 GB machine, so requests queue in arrival order and the queue is bounded. Streaming runs through a per-request channel from the worker to the connection's thread, and the connection's thread cancels the job when its client has gone, which the engine honours before the next decode step. The synthetic model drives the whole server in the default test suite, over a real socket, in every preset including the sanitizers, which is how the thread sanitizer found that the tokenizer's merge lookup was not safe to call from two threads at once.


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

`scripts/check_all.sh` runs every gate in one go: format check, build and tests for the debug, release, ASan+UBSan and TSan presets, clang-tidy, the integration tests when models are present, and a benchmark smoke run. It is the required check before a change is merged. The server and its two header-only libraries are built unless `AXON_BUILD_SERVER` is off.

Presets: `debug`, `release`, `release-native` (adds `-march=native`), `asan-ubsan`, `tsan`, `fuzz`, `coverage`, `tidy`.

Dependencies are fetched by CMake at configure time: fmt, CLI11, Catch2 (tests), Google Benchmark (bench), cpp-httplib and nlohmann/json (server). Nothing else.

Tests that need real model files are opt-in: run `scripts/download_models.sh`, then `AXON_MODELS_DIR=$PWD/models ctest --preset release -L integration` (an absolute path, because ctest runs from the build tree). Without the variable they are skipped. The integration tests take under four minutes with the default thread count, most of it the F16 perplexity pass.

Fixtures under `tests/reference/fixtures/` are generated by the Python scripts next to them and are small enough to commit:

- GGUF fixtures come from the official `gguf` writer, including a random two-layer LLaMA model in F32, Q8_0 and Q4_0.
- Tokenizer fixtures are tokenizer-only GGUF files plus the reference ids from Hugging Face.
- Kernel fixtures are `.axt` tensors (a 4-byte magic, dims, dtype, raw data) written by NumPy in float64.
- Model fixtures are a NumPy forward over the synthetic model, and for SmolLM2 the Hugging Face argmax per position, last-position logits and greedy continuations. The full per-position logits stay in the gitignored `tests/reference/local/` and are checked when `AXON_LOCAL_FIXTURES` points there.
- Chat template fixtures are the Hugging Face `apply_chat_template` output for three conversations per family, produced from the tokenizers alone.
- Benchmark results under `bench/results/` are written by `bench/bench_e2e.sh` and never edited. The `llamacpp` rows come from `scripts/compare_llamacpp.sh`, which builds llama.cpp at the commit the script pins and runs `llama-bench` with the same prompt and generation lengths. `bench/render_tables.py` renders them, and `--self-test` checks the renderer in the check script.
- Quantization fixtures are rows and small matrices quantized by the `gguf` package, with its dequantized values alongside.
- The perplexity text is the head of the WikiText-2 raw test split (Merity et al., "Pointer Sentinel Mixture Models", 2016, CC BY-SA 3.0), written by `scripts/make_perplexity_fixture.py`, with Hugging Face reference numbers for the SmolLM2 files from `tests/reference/gen_perplexity_reference.py`.

To add a kernel: declare it in `src/kernels/kernels.h` and `src/kernels/ref/ref.h`, implement the scalar version under `src/kernels/ref/`, register it in the table in `src/kernels/kernels.cpp`, add its reference to the generator and regenerate, then write a fixture test plus at least one property test under `tests/unit/`.

## License

MIT
