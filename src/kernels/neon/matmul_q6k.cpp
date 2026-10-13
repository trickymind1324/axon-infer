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

// Sixteen six-bit values, already combined into int8 lanes, times a float scale: the
// dequantized weights as four float vectors, equal to the scalar kernel's values element for
// element. They are computed once per weight block and applied to every activation row of a
// tile, which is where the multi-row kernel gets its time back.
struct Deq16 {
  float32x4_t v[4];
};

inline Deq16 dequant16(int8x16_t q, float scale) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(q));
  const int16x8_t hi = vmovl_s8(vget_high_s8(q));
  return {{vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), scale),
           vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), scale),
           vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), scale),
           vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), scale)}};
}

// One weight row of Q6_K super-blocks against R activation rows `x_stride` floats apart. The
// unpacking follows ggml's dequantize_row_q6_K element for element. Every activation row keeps
// its own four accumulators and receives the same fused multiply-adds in the same order
// whatever R is, so a row's result does not depend on how many rows share the call. Only the
// summation order differs from the scalar kernel, which the tests bound.
template <std::size_t R>
void dot_q6_k_tile(const BlockQ6_K* row, const float* x, std::size_t x_stride, std::size_t n_blocks,
                   float* out, std::size_t out_stride) {
  float32x4_t acc[R][4];
  for (std::size_t r = 0; r < R; ++r) {
    for (auto& a : acc[r]) {
      a = vdupq_n_f32(0.0F);
    }
  }
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);
  const uint8x16_t two_bits = vdupq_n_u8(0x03);
  const int8x16_t thirty_two = vdupq_n_s8(32);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float d = half(row[b].d);
    const std::uint8_t* ql = row[b].ql;
    const std::uint8_t* qh = row[b].qh;
    const std::int8_t* sc = row[b].scales;
    const std::size_t block_offset = b * kQK_K;
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
        const Deq16 w1 = dequant16(q1, d * static_cast<float>(sc[is]));
        const Deq16 w2 = dequant16(q2, d * static_cast<float>(sc[is + 2]));
        const Deq16 w3 = dequant16(q3, d * static_cast<float>(sc[is + 4]));
        const Deq16 w4 = dequant16(q4, d * static_cast<float>(sc[is + 6]));
        const std::size_t base = block_offset + n + l;
        for (std::size_t r = 0; r < R; ++r) {
          const float* xr = x + r * x_stride + base;
          for (std::size_t g = 0; g < 4; ++g) {
            acc[r][0] = vfmaq_f32(acc[r][0], w1.v[g], vld1q_f32(xr + 4 * g));
          }
          for (std::size_t g = 0; g < 4; ++g) {
            acc[r][1] = vfmaq_f32(acc[r][1], w2.v[g], vld1q_f32(xr + 32 + 4 * g));
          }
          for (std::size_t g = 0; g < 4; ++g) {
            acc[r][2] = vfmaq_f32(acc[r][2], w3.v[g], vld1q_f32(xr + 64 + 4 * g));
          }
          for (std::size_t g = 0; g < 4; ++g) {
            acc[r][3] = vfmaq_f32(acc[r][3], w4.v[g], vld1q_f32(xr + 96 + 4 * g));
          }
        }
      }
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
  for (std::size_t r = 0; r < R; ++r) {
    out[r * out_stride] =
        vaddvq_f32(vaddq_f32(vaddq_f32(acc[r][0], acc[r][1]), vaddq_f32(acc[r][2], acc[r][3])));
  }
}

std::size_t super_blocks_of(std::size_t n_in) {
  AXON_DCHECK(n_in % kQK_K == 0, "Q6_K matmul: n_in {} is not a multiple of {}", n_in, kQK_K);
  return n_in / kQK_K;
}

void matmul_q6_k(const BlockQ6_K* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  const std::size_t nb = super_blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    dot_q6_k_tile<1>(w + o * nb, x, 0, nb, y + o, 0);
  }
}

// Activation rows in groups of four, each group against every weight row, so a group's rows
// stay in the first-level cache while the weights stream past. The tail rows and the single-row
// kernel are the same template with R equal to one.
constexpr std::size_t kRowTile = 4;

void matmul_q6_k_rows(const BlockQ6_K* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  const std::size_t nb = super_blocks_of(n_in);
  std::size_t r = 0;
  for (; r + kRowTile <= n_rows; r += kRowTile) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_q6_k_tile<kRowTile>(w + o * nb, x + r * n_in, n_in, nb, y + r * y_stride + o, y_stride);
    }
  }
  for (; r < n_rows; ++r) {
    for (std::size_t o = 0; o < n_out; ++o) {
      dot_q6_k_tile<1>(w + o * nb, x + r * n_in, 0, nb, y + r * y_stride + o, 0);
    }
  }
}

}  // namespace

void install_q6k(KernelTable& table) {
  table.matmul_q6_k = matmul_q6_k;
  table.matmul_q6_k_rows = matmul_q6_k_rows;
}

}  // namespace axon::kernels::neon
