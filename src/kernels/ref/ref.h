#pragma once

// Scalar reference implementations. Plain loops, no intrinsics, written to be read.

#include "kernels/kernels.h"

#include <cstddef>
#include <cstdint>

namespace axon::kernels::ref {

void matmul_f32(const float* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
void matmul_f16(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                std::size_t n_in);
void matmul_bf16(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in);
void matmul_f32_rows(const float* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in);
void matmul_f16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in);
void matmul_bf16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in);

void rmsnorm(const float* x, const float* weight, float* y, std::size_t n, float eps);
void rope(float* x, std::size_t n_heads, std::size_t pos, const RopeParams& params);
void softmax(float* x, std::size_t n);
void silu_mul(const float* gate, const float* up, float* out, std::size_t n);
void embedding_row(DType type, const void* table, std::size_t n, std::int64_t index, float* out);
void attention(const float* q, const float* k, const float* v, float* out, float* scores,
               std::size_t seq, std::size_t n_head, std::size_t n_head_kv, std::size_t head_dim,
               float scale);
void attention_kv(const float* q, const float* k_cache, const float* v_cache, float* out,
                  float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                  std::size_t n_head_kv, std::size_t head_dim, float scale);
void attention_kv_heads(const float* q, const float* k_cache, const float* v_cache, float* out,
                        float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                        std::size_t n_head_kv, std::size_t head_dim, float scale,
                        std::size_t head_begin, std::size_t head_end);

void matmul_q8_0(const BlockQ8_0* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
void matmul_q4_0(const BlockQ4_0* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
void matmul_q8_0_q8(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in);
void matmul_q4_0_q8(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in);
void matmul_q8_0_rows(const BlockQ8_0* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in);
void matmul_q4_0_rows(const BlockQ4_0* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in);
void matmul_q8_0_q8_rows(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in);
void matmul_q4_0_q8_rows(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in);
void quantize_row_q8_0(const float* x, BlockQ8_0* out, std::size_t n);
void matmul_q4_1(const BlockQ4_1* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
void matmul_q4_1_q8(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in);
void matmul_q4_1_rows(const BlockQ4_1* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in);
void matmul_q4_1_q8_rows(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in);
void matmul_q6_k(const BlockQ6_K* w, const float* x, float* y, std::size_t n_out, std::size_t n_in);
void matmul_q6_k_rows(const BlockQ6_K* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in);

}  // namespace axon::kernels::ref
