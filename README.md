# axon-infer

LLM inference engine in C++20, built from scratch. Loads GGUF models from the LLaMA family and runs them on CPU with quantized weights, a KV cache, hand-written SIMD kernels, and a custom thread pool. No ggml, BLAS, or OpenMP. Every optimization is benchmarked so you can see what each one buys.

## Why

LLM inference is memory-bandwidth-bound on most hardware, not compute-bound. Production engines make that hard to see because they are hundreds of thousands of lines. axon-infer is a compact, tested engine where the whole path from file bytes to sampled token is readable, and where the benchmark table doubles as the explanation.

## Goals

- Load and run LLaMA-architecture models (SmolLM2, Qwen2.5, TinyLlama, Llama 3.2) from GGUF.
- Match a reference implementation's logits within numerical tolerance.
- Support FP32, FP16, Q8_0, and Q4_0 weights, dequantized inside the matmul kernel.
- Reach a competitive fraction of llama.cpp throughput on the same model and hardware, with a benchmark row for every optimization step.
- Ship an OpenAI-compatible HTTP server.

## Non-goals

Training, multimodal models, architectures outside the RMSNorm + RoPE + GQA + SwiGLU block, and beating llama.cpp. The goal is a clean, measured, understandable engine.

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

Layers underneath: `core` (arena allocator, thread pool, GGUF reader, dtypes), `kernels` (scalar reference plus NEON and AVX2 variants selected at runtime), `model` (LLaMA block, KV cache), `tokenizer`, `sampling`, `engine` (generation loop, batching).

Design rules: weights are memory-mapped and never copied. Activations come from one arena sized at load. The decode loop never allocates on the heap. Every kernel has a scalar reference and a test against it before any SIMD version exists.

## Benchmarks

Numbers are produced by the scripts in `bench/` and committed as JSON under `bench/results/`. Each row reports CPU, threads, model file, quantization, prompt length, generated tokens, and the median of five runs after one warm-up.

| Step | Model | Prefill tok/s | Generate tok/s | vs llama.cpp |
|---|---|---|---|---|
| Scalar, 1 thread, no cache | SmolLM2-135M F16 | 4.1 | 0.28 | _not yet measured_ |
| Scalar, 1 thread, KV cache | SmolLM2-135M F16 | 4.1 | 4.3 | _not yet measured_ |

The first row is the deliberately naive baseline: every generated token recomputes the whole sequence on the scalar kernels. The KV cache turns generation into one forward pass per token, so decode runs at the same rate as prefill. Every later optimization is measured against these rows.

## Quickstart

Requirements: CMake 3.24+, Ninja, a C++20 compiler (clang 17+, gcc 13+, or Apple clang 15+).

```sh
cmake --preset release
cmake --build --preset release
./build/release/apps/axon/axon --version

scripts/download_models.sh                     # SmolLM2-135M in F16, Q8_0 and Q4_0, checksums verified
./build/release/apps/axon-dump/axon-dump models/SmolLM2-135M-Instruct-Q8_0.gguf --no-arrays
```

`axon-dump` prints the header, every metadata key, the tensor table and a per-dtype byte summary. It reads any GGUF v3 file, including K-quant models it cannot run, and `--json` gives the same data as a document:

```
version 3  alignment 32  tensors 272  metadata 37  size 138.1 MB  data offset 1785952
architecture llama  name "Smollm2 135M 8k Lc100K Mix1 Ep2"
...
summary
  F32         61 tensors        140544 bytes   137.2 KB    0.1%
  Q8_0       211 tensors     142884864 bytes   136.3 MB   99.9%
  total      272 tensors     143025408 bytes   136.4 MB
```

`axon tokenize` encodes text with a model's own tokenizer and shows every token:

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

Two vocabulary models are supported, built entirely from GGUF metadata: byte-level BPE (`gpt2`) with the GPT-2, Llama 3 and Qwen2 pre-tokenizer patterns, and SentencePiece BPE (`llama`) with byte fallback. Output matches the Hugging Face tokenizers token for token on a 170-line corpus covering contractions, numbers, whitespace, code, emoji and a dozen scripts, for SmolLM2, TinyLlama, Llama 3.2 and Qwen2.5. Qwen's NFC normalization is implemented from generated Unicode tables, and a streaming decoder never emits a partial UTF-8 sequence.

`axon generate` runs greedy decoding: the prompt is prefilled into a KV cache and each new token is one forward pass over a single position. The output is exactly what Hugging Face produces:

```
$ ./build/release/apps/axon/axon generate -m models/SmolLM2-135M-Instruct-f16.gguf -p "The capital of France is" -n 20
model: Smollm2 135M 8k Lc100K Mix1 Ep2 (llama layers 30 embd 576 heads 9 kv 3 ff 1536 vocab 49152 ctx 8192 rope normal 100000 tied)
prompt: 5 tokens
The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace
prefill 5 tokens in 1.28 s (3.9 tok/s), decode 20 tokens in 4.70 s (4.26 tok/s), load 0.02 s, arena 1.0 MB, cache 1.1 MB of 90.0 MB
```

`--ctx` sets the cache length (default 2048, clamped to the model's training context) and `--no-cache` runs the cache-free loop that recomputes the whole sequence each step, which is also the path that can dump activations.

### Forward pass

`LlamaModel` reads its configuration from the GGUF metadata (`llama` and `qwen2` architectures), binds every weight with an exact shape and dtype check, and runs the block in the order the paper gives it: RMSNorm, Q/K/V projections, RoPE, causal grouped-query attention, output projection, RMSNorm, SwiGLU feed-forward, each with a residual add. Weights stay memory-mapped. Activations come from one arena sized at load, and a test asserts that the computed size equals the measured high-water mark and that a forward pass performs no heap allocation.

### KV cache

`KVCache` holds keys and values for every layer, preallocated for the cache length and filled in position order. The storage is an anonymous mapping, so context that is never reached costs no memory. `forward_prefill` appends a chunk of tokens and `forward_one` appends a single token. Both run the same layer code as the cache-free pass, writing new keys and values straight into the cache rows and attending over everything cached so far.

The defining test is bit-identity: cached decoding must produce exactly the floats the cache-free forward produces, at every position, however the tokens are split between prefill chunks and single steps. That holds on the scalar path because no reduction is ever reordered, and the unit tests check it on the synthetic model in every preset while the integration tests check it on SmolLM2. Context overflow is an error, never a silent grow.

Decode cost per token with a cache is nearly flat in context depth on the scalar path, because the attention over cached positions is small next to the weight matmuls:

| Model | Cached positions | ms per decoded token |
|---|---|---|
| SmolLM2-135M | 32 | 252 |
| SmolLM2-135M | 512 | 248 |
| SmolLM2-135M | 2048 | 266 |

Correctness is checked two ways. A float64 NumPy forward over a committed two-layer synthetic model runs in every preset with no download. With models present, SmolLM2-135M is compared against Hugging Face `LlamaForCausalLM` loaded from the same GGUF file: the argmax matches at every position of ten prompts, the logits agree to better than 5e-5 on a range of about 46, and 20-token greedy continuations match for five prompts. `--dump-layer N --dump-dir D` writes one layer's activations, `gen_model_fixtures.py --dump-hf N` writes the same tensors from Hugging Face hooks, and `scripts/compare_dumps.py` prints the difference per tensor.

Single-thread scalar forward on the M1, F16 weights:

| Model | Sequence | ms per token |
|---|---|---|
| SmolLM2-135M | 1 | 249 |
| SmolLM2-135M | 32 | 243 |
| SmolLM2-135M | 128 | 243 |

### Kernels

Every operation the forward pass needs lives behind one table of function pointers: matmul for F32, F16 and BF16 weights (single vector and multi-row), RMSNorm, RoPE (adjacent-pair and NeoX layouts, with the Llama 3.x per-dimension frequency factors), softmax, SwiGLU, embedding lookup and full-sequence causal attention with grouped-query heads. The scalar implementations under `src/kernels/ref/` are the reference: plain loops, sequential float accumulation, written to be read. Each has a fixture test computed in float64 by NumPy, and the RoPE and attention fixtures are cross-checked against PyTorch so the conventions are pinned before any model code exists.

Single-thread scalar matmul on an Apple M1, the baseline every later kernel is measured against:

| Weights | Shape (out x in) | GFLOP/s | GB/s |
|---|---|---|---|
| F32 | 1536 x 576 | 2.6 | 5.2 |
| F32 | 2048 x 2048 | 2.1 | 4.3 |
| F32 | 8192 x 2048 | 2.2 | 4.4 |
| F16 | 1536 x 576 | 1.1 | 1.1 |
| F16 | 2048 x 2048 | 1.1 | 1.1 |
| F16 | 8192 x 2048 | 1.1 | 1.1 |

## Development

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset asan-ubsan && cmake --build --preset asan-ubsan && ctest --preset asan-ubsan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
./build/release/bench/bench_kernels
scripts/format.sh          # clang-format all sources
cmake --preset tidy && cmake --build --preset tidy
```

`scripts/check_all.sh` runs every gate in one go: format check, build and tests for the debug, release, ASan+UBSan and TSan presets, clang-tidy, and a benchmark smoke run. It is the required check before a change is merged.

Presets: `debug`, `release`, `release-native` (adds `-march=native`), `asan-ubsan`, `tsan`, `fuzz`, `coverage`, `tidy`. Default release builds use runtime CPU dispatch so one binary runs on any x86-64 or arm64 machine.

Dependencies are fetched by CMake at configure time: fmt, CLI11, Catch2 (tests), Google Benchmark (bench). Nothing else.

Tests that need real model files are opt-in: run `scripts/download_models.sh`, then `AXON_MODELS_DIR=$PWD/models ctest --preset release -L integration` (an absolute path, because ctest runs from the build tree). Without the variable they are skipped. Fixtures under `tests/reference/fixtures/` are generated by the Python scripts next to them using the official `gguf` package, and are small enough to commit, so the regular test suite never downloads anything.

Tokenizer fixtures are tokenizer-only GGUF files (metadata without tensors) plus the reference ids from Hugging Face, so the SmolLM2 and TinyLlama tests need no download. Llama 3.2 and Qwen2.5 are checked by the opt-in integration tests. `bench_tokenizer` reports encode throughput.

Kernel fixtures are `.axt` files (a 4-byte magic, dims, dtype, raw data) written by `tests/reference/gen_kernel_fixtures.py`. To add a kernel: declare it in `src/kernels/kernels.h` and `src/kernels/ref/ref.h`, implement the scalar version under `src/kernels/ref/`, register it in the table in `src/kernels/kernels.cpp`, add its reference to the generator and regenerate, then write a fixture test plus at least one property test under `tests/unit/`. Optimized variants come later as separate tables checked against `scalar_kernels()`.

Model fixtures come from `tests/reference/gen_model_fixtures.py`: the synthetic logits by default, and the SmolLM2 references with `--smollm2 models/SmolLM2-135M-Instruct-f16.gguf` (needs `torch`, `transformers` and `accelerate` in the venv). The committed SmolLM2 set is about 2 MB. The full per-position logits land in the gitignored `tests/reference/local/` and are checked by the integration tests when `AXON_LOCAL_FIXTURES` points there. The SmolLM2 integration tests take about three minutes on the scalar path.

The GGUF parser is fuzzed two ways: a deterministic corruption replay runs inside the normal unit tests under every preset, and a libFuzzer target builds with the `fuzz` preset on Linux clang.

## License

MIT
