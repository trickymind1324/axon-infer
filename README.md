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

Design rules: weights are memory-mapped and never copied; activations come from one arena sized at load; no heap allocation in the decode loop; every kernel has a scalar reference and a test against it before any SIMD version exists.

## Quickstart

Requirements: CMake 3.24+, Ninja, a C++20 compiler (clang 17+, gcc 13+, or Apple clang 15+).

```sh
cmake --preset release
cmake --build --preset release
./build/release/apps/axon/axon --version
```

## Development

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset asan-ubsan && cmake --build --preset asan-ubsan && ctest --preset asan-ubsan
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan
./build/release/bench/bench_kernels
scripts/format.sh          # clang-format all sources
cmake --preset tidy && cmake --build --preset tidy
```

Presets: `debug`, `release`, `release-native` (adds `-march=native`), `asan-ubsan`, `tsan`, `coverage`, `tidy`. Default release builds use runtime CPU dispatch so one binary runs on any x86-64 or arm64 machine.

Dependencies are fetched by CMake at configure time: fmt, CLI11, Catch2 (tests), Google Benchmark (bench). Nothing else.

## License

MIT
