#include "core/check.h"

#include "kernels/neon/neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace axon::kernels::neon {

namespace {

// The dot product of two 32-byte int8 blocks as an exact int32. Lanes accumulate with
// vdotq_s32 and are summed once per block, which is exact whatever the lane order, so the
// scalar kernel's float scaling and accumulation across blocks can be kept unchanged and the
// result is bit-identical.
std::int32_t block_dot(const std::int8_t* a, const std::int8_t* b) {
  int32x4_t acc = vdupq_n_s32(0);
  acc = vdotq_s32(acc, vld1q_s8(a), vld1q_s8(b));
  acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
  return vaddvq_s32(acc);
}

// Q4_0 nibbles as signed int8 lanes: elements 0..15 are the low nibbles minus 8, elements
// 16..31 the high nibbles minus 8, the layout the scalar kernel and ggml use.
std::int32_t block_dot_q4_0(const std::uint8_t* nibbles, const std::int8_t* x) {
  const uint8x16_t packed = vld1q_u8(nibbles);
  const int8x16_t low =
      vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
  const int8x16_t high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), vdupq_n_s8(8));
  int32x4_t acc = vdupq_n_s32(0);
  acc = vdotq_s32(acc, low, vld1q_s8(x));
  acc = vdotq_s32(acc, high, vld1q_s8(x + 16));
  return vaddvq_s32(acc);
}

// Q4_1 nibbles are unsigned, and the block offset m multiplies the activation's integer sum.
void block_dot_q4_1(const std::uint8_t* nibbles, const std::int8_t* x, std::int32_t* sumi,
                    std::int32_t* sumx) {
  const uint8x16_t packed = vld1q_u8(nibbles);
  const int8x16_t low = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0F)));
  const int8x16_t high = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
  const int8x16_t x0 = vld1q_s8(x);
  const int8x16_t x1 = vld1q_s8(x + 16);
  int32x4_t acc = vdupq_n_s32(0);
  acc = vdotq_s32(acc, low, x0);
  acc = vdotq_s32(acc, high, x1);
  *sumi = vaddvq_s32(acc);
  *sumx = static_cast<std::int32_t>(vaddlvq_s8(x0)) + static_cast<std::int32_t>(vaddlvq_s8(x1));
}

// fp16 to fp32 through the conversion instruction, exact like the software conversion and
// inlined, where the function call was two per block.
inline float half(std::uint16_t bits) {
  return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(bits))), 0);
}

std::size_t blocks_of(std::size_t n_in) {
  AXON_DCHECK(n_in % kQK8_0 == 0, "quantized matmul: n_in {} is not a multiple of {}", n_in,
              kQK8_0);
  return n_in / kQK8_0;
}

// Four output rows per pass so the per-row float chains overlap. Each row's accumulation is
// still block by block in order, which is what keeps the result bit-identical to scalar.
void matmul_q8_0_q8(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  std::size_t o = 0;
  for (; o + 4 <= n_out; o += 4) {
    const BlockQ8_0* r0 = w + o * nb;
    const BlockQ8_0* r1 = r0 + nb;
    const BlockQ8_0* r2 = r1 + nb;
    const BlockQ8_0* r3 = r2 + nb;
    float a0 = 0.0F;
    float a1 = 0.0F;
    float a2 = 0.0F;
    float a3 = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const float dx = half(xq[b].d);
      a0 += static_cast<float>(block_dot(r0[b].qs, xq[b].qs)) * half(r0[b].d) * dx;
      a1 += static_cast<float>(block_dot(r1[b].qs, xq[b].qs)) * half(r1[b].d) * dx;
      a2 += static_cast<float>(block_dot(r2[b].qs, xq[b].qs)) * half(r2[b].d) * dx;
      a3 += static_cast<float>(block_dot(r3[b].qs, xq[b].qs)) * half(r3[b].d) * dx;
    }
    y[o] = a0;
    y[o + 1] = a1;
    y[o + 2] = a2;
    y[o + 3] = a3;
  }
  for (; o < n_out; ++o) {
    const BlockQ8_0* row = w + o * nb;
    float acc = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      acc += static_cast<float>(block_dot(row[b].qs, xq[b].qs)) * half(row[b].d) * half(xq[b].d);
    }
    y[o] = acc;
  }
}

void matmul_q4_0_q8(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  std::size_t o = 0;
  for (; o + 4 <= n_out; o += 4) {
    const BlockQ4_0* r0 = w + o * nb;
    const BlockQ4_0* r1 = r0 + nb;
    const BlockQ4_0* r2 = r1 + nb;
    const BlockQ4_0* r3 = r2 + nb;
    float a0 = 0.0F;
    float a1 = 0.0F;
    float a2 = 0.0F;
    float a3 = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      const float dx = half(xq[b].d);
      a0 += static_cast<float>(block_dot_q4_0(r0[b].qs, xq[b].qs)) * half(r0[b].d) * dx;
      a1 += static_cast<float>(block_dot_q4_0(r1[b].qs, xq[b].qs)) * half(r1[b].d) * dx;
      a2 += static_cast<float>(block_dot_q4_0(r2[b].qs, xq[b].qs)) * half(r2[b].d) * dx;
      a3 += static_cast<float>(block_dot_q4_0(r3[b].qs, xq[b].qs)) * half(r3[b].d) * dx;
    }
    y[o] = a0;
    y[o + 1] = a1;
    y[o + 2] = a2;
    y[o + 3] = a3;
  }
  for (; o < n_out; ++o) {
    const BlockQ4_0* row = w + o * nb;
    float acc = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      acc +=
          static_cast<float>(block_dot_q4_0(row[b].qs, xq[b].qs)) * half(row[b].d) * half(xq[b].d);
    }
    y[o] = acc;
  }
}

void matmul_q4_1_q8(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_out,
                    std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t o = 0; o < n_out; ++o) {
    const BlockQ4_1* row = w + o * nb;
    float acc = 0.0F;
    for (std::size_t b = 0; b < nb; ++b) {
      std::int32_t sumi = 0;
      std::int32_t sumx = 0;
      block_dot_q4_1(row[b].qs, xq[b].qs, &sumi, &sumx);
      const float dx = half(xq[b].d);
      acc += static_cast<float>(sumi) * half(row[b].d) * dx +
             half(row[b].m) * dx * static_cast<float>(sumx);
    }
    y[o] = acc;
  }
}

void matmul_q8_0_q8_rows(const BlockQ8_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q8_0_q8(w, xq + r * nb, y + r * n_out, n_out, n_in);
  }
}

void matmul_q4_0_q8_rows(const BlockQ4_0* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_0_q8(w, xq + r * nb, y + r * n_out, n_out, n_in);
  }
}

void matmul_q4_1_q8_rows(const BlockQ4_1* w, const BlockQ8_0* xq, float* y, std::size_t n_rows,
                         std::size_t n_out, std::size_t n_in) {
  const std::size_t nb = blocks_of(n_in);
  for (std::size_t r = 0; r < n_rows; ++r) {
    matmul_q4_1_q8(w, xq + r * nb, y + r * n_out, n_out, n_in);
  }
}

// ggml's quantize_row_q8_0 with vector absolute maximum and vcvtaq, which rounds to nearest
// with ties away from zero exactly as roundf does, so the bytes match the scalar kernel.
void quantize_row_q8_0(const float* x, BlockQ8_0* out, std::size_t n) {
  AXON_DCHECK(n % kQK8_0 == 0, "quantize_row_q8_0: {} is not a multiple of {}", n, kQK8_0);
  for (std::size_t b = 0; b < n / kQK8_0; ++b) {
    const float* block = x + b * kQK8_0;
    float32x4_t vmax = vdupq_n_f32(0.0F);
    for (std::uint32_t i = 0; i < kQK8_0; i += 4) {
      vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(block + i)));
    }
    const float amax = vmaxvq_f32(vmax);
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    out[b].d = fp32_to_fp16(d);
    for (std::uint32_t i = 0; i < kQK8_0; i += 8) {
      const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(block + i), id));
      const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(vld1q_f32(block + i + 4), id));
      const int16x8_t q16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
      vst1_s8(out[b].qs + i, vqmovn_s16(q16));
    }
  }
}

}  // namespace

void install_quant(KernelTable& table) {
  table.matmul_q8_0_q8 = matmul_q8_0_q8;
  table.matmul_q4_0_q8 = matmul_q4_0_q8;
  table.matmul_q4_1_q8 = matmul_q4_1_q8;
  table.matmul_q8_0_q8_rows = matmul_q8_0_q8_rows;
  table.matmul_q4_0_q8_rows = matmul_q4_0_q8_rows;
  table.matmul_q4_1_q8_rows = matmul_q4_1_q8_rows;
  table.quantize_row_q8_0 = quantize_row_q8_0;
}

}  // namespace axon::kernels::neon
