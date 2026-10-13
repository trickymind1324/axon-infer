#pragma once

// Numerical kernels for the LLaMA forward pass, reached through a table of function pointers so
// SIMD variants can replace entries later without touching any call site.
//
// Conventions: everything is row-major and contiguous. A weight W for y = W x is stored as
// [n_out][n_in], which is exactly the GGUF layout (shape {n_in, n_out}, innermost first), so no
// transposition ever happens. Reductions run sequentially in float; the scalar table is the
// reduction order that deterministic mode will pin.

#include "core/dtype.h"
#include "core/expected.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace axon::kernels {

// Normal rotates adjacent pairs (x[2i], x[2i+1]) as Llama 2 and Llama 3 do once the GGUF
// converter has permuted their Q/K weights. NeoX rotates halves (x[i], x[i + rotary_dim/2]) as
// Qwen2 does.
enum class RopeStyle { Normal, NeoX };

struct RopeParams {
  std::size_t head_dim;
  std::size_t rotary_dim;  // even, at most head_dim; equal to head_dim for every target model
  float theta;             // rope_freq_base
  RopeStyle style;
  const float* freq_factors;  // nullptr, or rotary_dim/2 divisors of inv_freq (Llama 3.x scaling)
};

struct KernelTable {
  // y[n_out] = W[n_out][n_in] x[n_in]
  void (*matmul_f32)(const float* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
  void (*matmul_f16)(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                     std::size_t n_in);
  void (*matmul_bf16)(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                      std::size_t n_in);
  // Y[n_rows][n_out] = X[n_rows][n_in] W^T (prefill)
  void (*matmul_f32_rows)(const float* w, const float* x, float* y, std::size_t n_rows,
                          std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_f16_rows)(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                          std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_bf16_rows)(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                           std::size_t n_out, std::size_t n_in, std::size_t y_stride);

  // y = x / sqrt(mean(x^2) + eps) * weight
  void (*rmsnorm)(const float* x, const float* weight, float* y, std::size_t n, float eps);
  // In place on x[n_heads][head_dim] for one position.
  void (*rope)(float* x, std::size_t n_heads, std::size_t pos, const RopeParams& params);
  // In place, max-subtracted so any input range is safe.
  void (*softmax)(float* x, std::size_t n);
  // out = silu(gate) * up
  void (*silu_mul)(const float* gate, const float* up, float* out, std::size_t n);
  // out[n] = row `index` of a [rows][n] table of the given type, converted to float.
  void (*embedding_row)(DType type, const void* table, std::size_t n, std::int64_t index,
                        float* out);
  // Full-sequence causal attention with grouped-query heads (head h reads kv head
  // h / (n_head / n_head_kv)).
  //   q: [seq][n_head][head_dim]   k, v: [seq][n_head_kv][head_dim]   out: [seq][n_head][head_dim]
  //   scores: scratch of at least seq * seq floats
  void (*attention)(const float* q, const float* k, const float* v, float* out, float* scores,
                    std::size_t seq, std::size_t n_head, std::size_t n_head_kv,
                    std::size_t head_dim, float scale);
  // Attention over a key/value cache. Queries sit at absolute positions n_past .. n_past + n_q - 1
  // and attend causally over cached positions 0 .. n_past + n_q - 1, whose keys and values the
  // caller has already written. With n_past == 0 and n_q == seq this is `attention`.
  //   q: [n_q][n_head][head_dim]   k_cache, v_cache: [n_past + n_q][n_head_kv][head_dim]
  //   out: [n_q][n_head][head_dim]   scores: scratch of at least n_q * (n_past + n_q) floats
  void (*attention_kv)(const float* q, const float* k_cache, const float* v_cache, float* out,
                       float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                       std::size_t n_head_kv, std::size_t head_dim, float scale);
  // The same over heads [head_begin, head_end) only, so threads can split the heads with a
  // score scratch each. attention_kv is this with the full range.
  void (*attention_kv_heads)(const float* q, const float* k_cache, const float* v_cache, float* out,
                             float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                             std::size_t n_head_kv, std::size_t head_dim, float scale,
                             std::size_t head_begin, std::size_t head_end);
  // The same over a cache stored as fp16 bit patterns, widened as they are read.
  void (*attention_kv_heads_f16)(const float* q, const std::uint16_t* k_cache,
                                 const std::uint16_t* v_cache, float* out, float* scores,
                                 std::size_t n_q, std::size_t n_past, std::size_t n_head,
                                 std::size_t n_head_kv, std::size_t head_dim, float scale,
                                 std::size_t head_begin, std::size_t head_end);

  // Quantized weights, F32 activations: y[o] = sum_i dequant(w[o][i]) * x[i], sequential in
  // float. n_in is a multiple of 32.
  void (*matmul_q8_0)(const BlockQ8_0* w, const float* x, float* y, std::size_t n_out,
                      std::size_t n_in);
  void (*matmul_q4_0)(const BlockQ4_0* w, const float* x, float* y, std::size_t n_out,
                      std::size_t n_in);
  // Quantized weights, Q8_0 activations: per block an int32 dot of the two int8 vectors,
  // scaled by both fp16 scales and accumulated in float. The reduction the SIMD kernels
  // reproduce.
  void (*matmul_q8_0_q8)(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                         std::size_t n_in);
  void (*matmul_q4_0_q8)(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                         std::size_t n_in);
  // Rows variants: Y[n_rows][n_out] from X[n_rows][n_in] (float rows or Q8_0 block rows).
  void (*matmul_q8_0_rows)(const BlockQ8_0* w, const float* x, float* y, std::size_t n_rows,
                           std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_q4_0_rows)(const BlockQ4_0* w, const float* x, float* y, std::size_t n_rows,
                           std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_q8_0_q8_rows)(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                              std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_q4_0_q8_rows)(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                              std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  // Activation quantization, one row of n floats (a multiple of 32) into n / 32 blocks.
  void (*quantize_row_q8_0)(const float* x, BlockQ8_0* out, std::size_t n);

  // Q4_1 (scale and offset per block), both activation paths. With Q8_0 activations the offset
  // contributes m * d_x * sum(q_x) per block.
  void (*matmul_q4_1)(const BlockQ4_1* w, const float* x, float* y, std::size_t n_out,
                      std::size_t n_in);
  void (*matmul_q4_1_q8)(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                         std::size_t n_in);
  void (*matmul_q4_1_rows)(const BlockQ4_1* w, const float* x, float* y, std::size_t n_rows,
                           std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  void (*matmul_q4_1_q8_rows)(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                              std::size_t n_out, std::size_t n_in, std::size_t y_stride);
  // Q6_K with float activations only: each 256-element super-block is dequantized and dotted.
  // Used for output and embedding matrices, n_in a multiple of 256.
  void (*matmul_q6_k)(const BlockQ6_K* w, const float* x, float* y, std::size_t n_out,
                      std::size_t n_in);
  void (*matmul_q6_k_rows)(const BlockQ6_K* w, const float* x, float* y, std::size_t n_rows,
                           std::size_t n_out, std::size_t n_in, std::size_t y_stride);
};

// The scalar reference table, the baseline every optimized table is tested against and the
// table the kernel fixture tests exercise.
const KernelTable& kernels();
const KernelTable& scalar_kernels();

enum class KernelSet : std::uint8_t { Auto, Scalar, Neon };

// The table for a set. Auto picks the best the host supports: Neon on an arm64 machine with
// the dot product extension, Scalar otherwise. Asking for Neon on a host without it is an
// error, never a silent fallback.
Expected<const KernelTable*> select(KernelSet set);
// "scalar" or "neon", for banners and benchmark records.
std::string_view name(const KernelTable& table);
// Parses "auto", "scalar" or "neon".
bool parse_kernel_set(std::string_view text, KernelSet& out);

}  // namespace axon::kernels
