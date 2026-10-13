#include "core/check.h"
#include "core/dtype.h"

#include "kernels/ref/ref.h"

#include <cstddef>
#include <cstdint>

namespace axon::kernels::ref {

namespace {

// One weight row against a float activation: each block is dequantized on the fly and the
// products accumulate left to right in float.
float dot_q8_0_f32(const BlockQ8_0* row, const float* x, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float d = fp16_to_fp32(row[b].d);
    const float* xb = x + b * kQK8_0;
    for (std::uint32_t i = 0; i < kQK8_0; ++i) {
      acc += static_cast<float>(row[b].qs[i]) * d * xb[i];
    }
  }
  return acc;
}

float dot_q4_0_f32(const BlockQ4_0* row, const float* x, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float d = fp16_to_fp32(row[b].d);
    const float* xb = x + b * kQK4_0;
    for (std::uint32_t i = 0; i < kQK4_0 / 2; ++i) {
      const auto low = static_cast<float>(static_cast<int>(row[b].qs[i] & 0x0F) - 8);
      const auto high = static_cast<float>(static_cast<int>(row[b].qs[i] >> 4) - 8);
      acc += low * d * xb[i];
      acc += high * d * xb[i + kQK4_0 / 2];
    }
  }
  return acc;
}

// One weight row against a Q8_0 activation: an exact int32 dot per block, scaled once by the
// product of the two block scales. This is ggml's reference reduction.
float dot_q8_0_q8(const BlockQ8_0* row, const BlockQ8_0* xq, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    std::int32_t sumi = 0;
    for (std::uint32_t i = 0; i < kQK8_0; ++i) {
      sumi += static_cast<std::int32_t>(row[b].qs[i]) * static_cast<std::int32_t>(xq[b].qs[i]);
    }
    acc += static_cast<float>(sumi) * fp16_to_fp32(row[b].d) * fp16_to_fp32(xq[b].d);
  }
  return acc;
}

float dot_q4_0_q8(const BlockQ4_0* row, const BlockQ8_0* xq, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    std::int32_t sumi = 0;
    for (std::uint32_t i = 0; i < kQK4_0 / 2; ++i) {
      const int low = static_cast<int>(row[b].qs[i] & 0x0F) - 8;
      const int high = static_cast<int>(row[b].qs[i] >> 4) - 8;
      sumi += low * static_cast<std::int32_t>(xq[b].qs[i]);
      sumi += high * static_cast<std::int32_t>(xq[b].qs[i + kQK4_0 / 2]);
    }
    acc += static_cast<float>(sumi) * fp16_to_fp32(row[b].d) * fp16_to_fp32(xq[b].d);
  }
  return acc;
}

std::size_t blocks_of(std::size_t n_in) {
  AXON_DCHECK(n_in % kQK8_0 == 0, "quantized matmul: n_in {} is not a multiple of {}", n_in,
              kQK8_0);
  return n_in / kQK8_0;
}

}  // namespace

void matmul_q8_0(const BlockQ8_0* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q8_0_f32(w + o * nb, x, nb);
  }
}

void matmul_q4_0(const BlockQ4_0* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q4_0_f32(w + o * nb, x, nb);
  }
}

void matmul_q8_0_q8(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q8_0_q8(w + o * nb, xq, nb);
  }
}

void matmul_q4_0_q8(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q4_0_q8(w + o * nb, xq, nb);
  }
}

void matmul_q8_0_rows(const BlockQ8_0* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q8_0(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

void matmul_q4_0_rows(const BlockQ4_0* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_0(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

void matmul_q8_0_q8_rows(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q8_0_q8(w, xq + r * nb, y + r * y_stride, n_out, n_in);
  }
}

void matmul_q4_0_q8_rows(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_0_q8(w, xq + r * nb, y + r * y_stride, n_out, n_in);
  }
}

void quantize_row_q8_0(const float* x, BlockQ8_0* out, std::size_t n) {
  axon::quantize_row_q8_0(x, out, n);
}

namespace {

float dot_q4_1_f32(const BlockQ4_1* row, const float* x, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float d = fp16_to_fp32(row[b].d);
    const float m = fp16_to_fp32(row[b].m);
    const float* xb = x + b * kQK4_1;
    for (std::uint32_t i = 0; i < kQK4_1 / 2; ++i) {
      acc += (static_cast<float>(row[b].qs[i] & 0x0F) * d + m) * xb[i];
      acc += (static_cast<float>(row[b].qs[i] >> 4) * d + m) * xb[i + kQK4_1 / 2];
    }
  }
  return acc;
}

// sum_i (q_i d + m) x_i over a block, with x_i = qx_i d_x, is
// d d_x sum(q qx) + m d_x sum(qx): one int dot plus the activation's integer sum.
float dot_q4_1_q8(const BlockQ4_1* row, const BlockQ8_0* xq, std::size_t n_blocks) {
  float acc = 0.0F;
  for (std::size_t b = 0; b < n_blocks; ++b) {
    std::int32_t sumi = 0;
    std::int32_t sumx = 0;
    for (std::uint32_t i = 0; i < kQK4_1 / 2; ++i) {
      const int low = static_cast<int>(row[b].qs[i] & 0x0F);
      const int high = static_cast<int>(row[b].qs[i] >> 4);
      sumi += low * static_cast<std::int32_t>(xq[b].qs[i]) +
              high * static_cast<std::int32_t>(xq[b].qs[i + kQK4_1 / 2]);
      sumx += static_cast<std::int32_t>(xq[b].qs[i]) +
              static_cast<std::int32_t>(xq[b].qs[i + kQK4_1 / 2]);
    }
    const float dx = fp16_to_fp32(xq[b].d);
    acc += static_cast<float>(sumi) * fp16_to_fp32(row[b].d) * dx +
           fp16_to_fp32(row[b].m) * dx * static_cast<float>(sumx);
  }
  return acc;
}

float dot_q6_k_f32(const BlockQ6_K* row, const float* x, std::size_t n_blocks) {
  float acc = 0.0F;
  float values[kQK_K];
  for (std::size_t b = 0; b < n_blocks; ++b) {
    dequantize_block_q6_k(row[b], values);
    const float* xb = x + b * kQK_K;
    for (std::uint32_t i = 0; i < kQK_K; ++i) {
      acc += values[i] * xb[i];
    }
  }
  return acc;
}

std::size_t super_blocks_of(std::size_t n_in) {
  AXON_DCHECK(n_in % kQK_K == 0, "Q6_K matmul: n_in {} is not a multiple of {}", n_in, kQK_K);
  return n_in / kQK_K;
}

}  // namespace

void matmul_q4_1(const BlockQ4_1* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q4_1_f32(w + o * nb, x, nb);
  }
}

void matmul_q4_1_q8(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q4_1_q8(w + o * nb, xq, nb);
  }
}

void matmul_q4_1_rows(const BlockQ4_1* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_1(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

void matmul_q4_1_q8_rows(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_1_q8(w, xq + r * nb, y + r * y_stride, n_out, n_in);
  }
}

void matmul_q6_k(const BlockQ6_K* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = super_blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q6_k_f32(w + o * nb, x, nb);
  }
}

void matmul_q6_k_rows(const BlockQ6_K* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q6_k(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

}  // namespace axon::kernels::ref
