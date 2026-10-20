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

// --- multi-row tiles -----------------------------------------------------------------------
//
// One weight row against R activation rows `n_in` floats apart. Every activation row keeps the
// accumulators and the fused multiply-add order of the single-row kernel above, so a row's
// result does not depend on how many rows share the call. What the tile saves is the weight
// side: each weight vector is loaded once, and for F16 widened once, per R rows.

template <std::size_t R>
void dot_f32_tile(const float* row, const float* x, std::size_t n, float* out,
                  std::size_t out_stride) {
  float32x4_t acc[R][4];
  for (auto& a : acc) {
    for (auto& v : a) {
      v = vdupq_n_f32(0.0F);
    }
  }
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const float32x4_t w0 = vld1q_f32(row + i);
    const float32x4_t w1 = vld1q_f32(row + i + 4);
    const float32x4_t w2 = vld1q_f32(row + i + 8);
    const float32x4_t w3 = vld1q_f32(row + i + 12);
    for (std::size_t r = 0; r < R; ++r) {
      const float* xr = x + r * n + i;
      acc[r][0] = vfmaq_f32(acc[r][0], w0, vld1q_f32(xr));
      acc[r][1] = vfmaq_f32(acc[r][1], w1, vld1q_f32(xr + 4));
      acc[r][2] = vfmaq_f32(acc[r][2], w2, vld1q_f32(xr + 8));
      acc[r][3] = vfmaq_f32(acc[r][3], w3, vld1q_f32(xr + 12));
    }
  }
  for (; i + 4 <= n; i += 4) {
    const float32x4_t w0 = vld1q_f32(row + i);
    for (std::size_t r = 0; r < R; ++r) {
      acc[r][0] = vfmaq_f32(acc[r][0], w0, vld1q_f32(x + r * n + i));
    }
  }
  for (std::size_t r = 0; r < R; ++r) {
    float sum =
        vaddvq_f32(vaddq_f32(vaddq_f32(acc[r][0], acc[r][1]), vaddq_f32(acc[r][2], acc[r][3])));
    for (std::size_t j = i; j < n; ++j) {
      sum += row[j] * x[r * n + j];
    }
    out[r * out_stride] = sum;
  }
}

template <std::size_t R>
void dot_f16_tile(const std::uint16_t* row, const float* x, std::size_t n, float* out,
                  std::size_t out_stride) {
  float32x4_t acc[R][2];
  for (auto& a : acc) {
    for (auto& v : a) {
      v = vdupq_n_f32(0.0F);
    }
  }
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t halves = vreinterpretq_f16_u16(vld1q_u16(row + i));
    const float32x4_t w0 = vcvt_f32_f16(vget_low_f16(halves));
    const float32x4_t w1 = vcvt_high_f32_f16(halves);
    for (std::size_t r = 0; r < R; ++r) {
      const float* xr = x + r * n + i;
      acc[r][0] = vfmaq_f32(acc[r][0], w0, vld1q_f32(xr));
      acc[r][1] = vfmaq_f32(acc[r][1], w1, vld1q_f32(xr + 4));
    }
  }
  for (std::size_t r = 0; r < R; ++r) {
    float sum = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
    for (std::size_t j = i; j < n; ++j) {
      sum += fp16_to_fp32(row[j]) * x[r * n + j];
    }
    out[r * out_stride] = sum;
  }
}

constexpr std::size_t kRowTileF32 = 4;
constexpr std::size_t kRowTileF16 = 8;

void matmul_f32_rows(const float* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  std::size_t r = 0;
  for (; r + kRowTileF32 <= n_rows; r += kRowTileF32) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_f32_tile<kRowTileF32>(w + o * n_in, x + r * n_in, n_in, y + r * y_stride + o, y_stride);
    }
  }
  if (r + 2 <= n_rows) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_f32_tile<2>(w + o * n_in, x + r * n_in, n_in, y + r * y_stride + o, y_stride);
    }
    r += 2;
  }
  for (; r < n_rows; ++r) {
    matmul_f32(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

void matmul_f16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  std::size_t r = 0;
  for (; r + kRowTileF16 <= n_rows; r += kRowTileF16) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_f16_tile<kRowTileF16>(w + o * n_in, x + r * n_in, n_in, y + r * y_stride + o, y_stride);
    }
  }
  // The tail descends through the smaller tiles, as the integer kernels do.
  if (r + 4 <= n_rows) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_f16_tile<4>(w + o * n_in, x + r * n_in, n_in, y + r * y_stride + o, y_stride);
    }
    r += 4;
  }
  if (r + 2 <= n_rows) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_f16_tile<2>(w + o * n_in, x + r * n_in, n_in, y + r * y_stride + o, y_stride);
    }
    r += 2;
  }
  for (; r < n_rows; ++r) {
    matmul_f16(w, x + r * n_in, y + r * y_stride, n_out, n_in);
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
