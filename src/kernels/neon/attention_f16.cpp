#include "core/check.h"

#include "kernels/neon/neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace axon::kernels::neon {

namespace {

// q[head_dim] dot an fp16 key row, widened eight at a time. Two accumulators, so the order
// differs from the scalar kernel within the tolerance the tests lock.
float dot_q_k16(const float* q, const std::uint16_t* k, std::size_t head_dim) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  std::size_t d = 0;
  for (; d + 8 <= head_dim; d += 8) {
    const float16x8_t halves = vreinterpretq_f16_u16(vld1q_u16(k + d));
    acc0 = vfmaq_f32(acc0, vld1q_f32(q + d), vcvt_f32_f16(vget_low_f16(halves)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(q + d + 4), vcvt_high_f32_f16(halves));
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; d < head_dim; ++d) {
    sum += q[d] * fp16_to_fp32(k[d]);
  }
  return sum;
}

// o[head_dim] += p * fp16 value row.
void axpy_v16(float* o, float p, const std::uint16_t* v, std::size_t head_dim) {
  const float32x4_t pv = vdupq_n_f32(p);
  std::size_t d = 0;
  for (; d + 8 <= head_dim; d += 8) {
    const float16x8_t halves = vreinterpretq_f16_u16(vld1q_u16(v + d));
    vst1q_f32(o + d, vfmaq_f32(vld1q_f32(o + d), pv, vcvt_f32_f16(vget_low_f16(halves))));
    vst1q_f32(o + d + 4, vfmaq_f32(vld1q_f32(o + d + 4), pv, vcvt_high_f32_f16(halves)));
  }
  for (; d < head_dim; ++d) {
    o[d] += p * fp16_to_fp32(v[d]);
  }
}

void attention_kv_heads_f16(const float* q, const std::uint16_t* k_cache,
                            const std::uint16_t* v_cache, float* out, float* scores,
                            std::size_t n_q, std::size_t n_past, std::size_t n_head,
                            std::size_t n_head_kv, std::size_t head_dim, float scale,
                            std::size_t head_begin, std::size_t head_end) {
  AXON_DCHECK(n_head_kv > 0 && n_head % n_head_kv == 0, "attention: {} heads over {} kv heads",
              n_head, n_head_kv);
  const std::size_t group = n_head / n_head_kv;
  const std::size_t total = n_past + n_q;
  const auto softmax = scalar_kernels().softmax;
  for (std::size_t h = head_begin; h < head_end; ++h) {
    const std::size_t kv_head = h / group;
    for (std::size_t i = 0; i < n_q; ++i) {
      const std::size_t pos = n_past + i;
      float* row = scores + i * total;
      const float* q_i = q + (i * n_head + h) * head_dim;
      for (std::size_t j = 0; j <= pos; ++j) {
        row[j] = scale * dot_q_k16(q_i, k_cache + (j * n_head_kv + kv_head) * head_dim, head_dim);
      }
      softmax(row, pos + 1);
      float* o = out + (i * n_head + h) * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        o[d] = 0.0F;
      }
      for (std::size_t j = 0; j <= pos; ++j) {
        axpy_v16(o, row[j], v_cache + (j * n_head_kv + kv_head) * head_dim, head_dim);
      }
    }
  }
}

}  // namespace

void install_attention_f16(KernelTable& table) {
  table.attention_kv_heads_f16 = attention_kv_heads_f16;
}

}  // namespace axon::kernels::neon
