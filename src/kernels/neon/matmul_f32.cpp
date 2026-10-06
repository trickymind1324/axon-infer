#include "kernels/neon/neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace axon::kernels::neon {

namespace {

// One output row as four independent vector accumulators over 16 elements per step, then a
// four-wide tail and a scalar tail. The reduction order differs from the scalar kernel, so the
// result is close rather than identical, within the tolerance the tests lock.
float dot_f32(const float* row, const float* x, std::size_t n) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(row + i + 8), vld1q_f32(x + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
  }
  for (; i + 4 <= n; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < n; ++i) {
    sum += row[i] * x[i];
  }
  return sum;
}

// Half-precision weights widened eight at a time. Converting is exact, so the only difference
// from the scalar kernel is again the summation order.
float dot_f16(const std::uint16_t* row, const float* x, std::size_t n) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t halves = vreinterpretq_f16_u16(vld1q_u16(row + i));
    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(halves)), vld1q_f32(x + i));
    acc1 = vfmaq_f32(acc1, vcvt_high_f32_f16(halves), vld1q_f32(x + i + 4));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) {
    sum += fp16_to_fp32(row[i]) * x[i];
  }
  return sum;
}

void matmul_f32(const float* w, const float* x, float* y, std::size_t n_out, std::size_t n_in) {
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_f32(w + o * n_in, x, n_in);
  }
}

void matmul_f16(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                std::size_t n_in) {
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_f16(w + o * n_in, x, n_in);
  }
}

void matmul_f32_rows(const float* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_f32(w, x + r * n_in, y + r * n_out, n_out, n_in);
  }
}

void matmul_f16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_f16(w, x + r * n_in, y + r * n_out, n_out, n_in);
  }
}

}  // namespace

void install_float(KernelTable& table) {
  table.matmul_f32 = matmul_f32;
  table.matmul_f16 = matmul_f16;
  table.matmul_f32_rows = matmul_f32_rows;
  table.matmul_f16_rows = matmul_f16_rows;
}

}  // namespace axon::kernels::neon
