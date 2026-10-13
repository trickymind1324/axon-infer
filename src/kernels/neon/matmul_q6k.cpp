#include "core/check.h"

#include "kernels/neon/neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace axon::kernels::neon {

namespace {

inline float half(std::uint16_t bits) {
  return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(bits))), 0);
}

// Sixteen six-bit values, already combined into int8 lanes, times a float scale, dotted with
// sixteen activations into the running accumulator.
inline float32x4_t fma16(float32x4_t acc, int8x16_t q, float scale, const float* x) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(q));
  const int16x8_t hi = vmovl_s8(vget_high_s8(q));
  acc =
      vfmaq_f32(acc, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), scale), vld1q_f32(x));
  acc = vfmaq_f32(acc, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scale),
                  vld1q_f32(x + 4));
  acc = vfmaq_f32(acc, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), scale),
                  vld1q_f32(x + 8));
  acc = vfmaq_f32(acc, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scale),
                  vld1q_f32(x + 12));
  return acc;
}

// One row of Q6_K super-blocks against float activations. The unpacking follows ggml's
// dequantize_row_q6_K element for element, so the dequantized values equal the scalar
// kernel's exactly, and only the summation order differs, which the tests bound.
float dot_q6_k(const BlockQ6_K* row, const float* x, std::size_t n_blocks) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);
  const uint8x16_t two_bits = vdupq_n_u8(0x03);
  const int8x16_t thirty_two = vdupq_n_s8(32);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float d = half(row[b].d);
    const std::uint8_t* ql = row[b].ql;
    const std::uint8_t* qh = row[b].qh;
    const std::int8_t* sc = row[b].scales;
    const float* xb = x + b * kQK_K;
    for (std::uint32_t n = 0; n < kQK_K; n += 128) {
      for (std::uint32_t l = 0; l < 32; l += 16) {
        const std::uint32_t is = l / 16;
        const uint8x16_t q_lo = vld1q_u8(ql + l);
        const uint8x16_t q_hi = vld1q_u8(ql + l + 32);
        const uint8x16_t h = vld1q_u8(qh + l);
        const int8x16_t q1 =
            vsubq_s8(vreinterpretq_s8_u8(
                         vorrq_u8(vandq_u8(q_lo, low_mask), vshlq_n_u8(vandq_u8(h, two_bits), 4))),
                     thirty_two);
        const int8x16_t q2 = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q_hi, low_mask),
                                         vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), two_bits), 4))),
            thirty_two);
        const int8x16_t q3 =
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(
                         vshrq_n_u8(q_lo, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), two_bits), 4))),
                     thirty_two);
        const int8x16_t q4 = vsubq_s8(
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q_hi, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4))),
            thirty_two);
        acc0 = fma16(acc0, q1, d * static_cast<float>(sc[is]), xb + n + l);
        acc1 = fma16(acc1, q2, d * static_cast<float>(sc[is + 2]), xb + n + l + 32);
        acc2 = fma16(acc2, q3, d * static_cast<float>(sc[is + 4]), xb + n + l + 64);
        acc3 = fma16(acc3, q4, d * static_cast<float>(sc[is + 6]), xb + n + l + 96);
      }
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

std::size_t super_blocks_of(std::size_t n_in) {
  AXON_DCHECK(n_in % kQK_K == 0, "Q6_K matmul: n_in {} is not a multiple of {}", n_in, kQK_K);
  return n_in / kQK_K;
}

void matmul_q6_k(const BlockQ6_K* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = super_blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    y[o] = dot_q6_k(w + o * nb, x, nb);
  }
}

void matmul_q6_k_rows(const BlockQ6_K* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q6_k(w, x + r * n_in, y + r * y_stride, n_out, n_in);
  }
}

}  // namespace

void install_q6k(KernelTable& table) {
  table.matmul_q6_k = matmul_q6_k;
  table.matmul_q6_k_rows = matmul_q6_k_rows;
}

}  // namespace axon::kernels::neon
