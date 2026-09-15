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
void softmax(float* x, std::size_t n);
void silu_mul(const float* gate, const float* up, float* out, std::size_t n);
void embedding_row(DType type, const void* table, std::size_t n, std::int64_t index, float* out);

}  // namespace axon::kernels::ref
