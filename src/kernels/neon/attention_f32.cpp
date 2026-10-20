#include "core/check.h"

#include "kernels/neon/neon.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace axon::kernels::neon {

namespace {

// q[head_dim] dot a key row, four accumulators over sixteen elements per step. The reduction
// order differs from the scalar kernel, within the tolerance the tests lock.
float dot_q_k(const float* q, const float* k, std::size_t head_dim) {
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  float32x4_t acc2 = vdupq_n_f32(0.0F);
  float32x4_t acc3 = vdupq_n_f32(0.0F);
  std::size_t d = 0;
  for (; d + 16 <= head_dim; d += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(q + d), vld1q_f32(k + d));
    acc1 = vfmaq_f32(acc1, vld1q_f32(q + d + 4), vld1q_f32(k + d + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(q + d + 8), vld1q_f32(k + d + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(q + d + 12), vld1q_f32(k + d + 12));
  }
  for (; d + 4 <= head_dim; d += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(q + d), vld1q_f32(k + d));
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; d < head_dim; ++d) {
    sum += q[d] * k[d];
  }
  return sum;
}

// o[d] = sum over j of p[j] * v[j][d], each output element a fused multiply-add chain in
// position order, which is what the scalar kernel's accumulation compiles to, so the output
// side is bit-identical to it. The chain stays in registers 32 elements at a time rather than
// being loaded and stored once per position.
void weighted_values(const float* p, const float* v_cache, std::size_t v_stride, std::size_t count,
                     float* o, std::size_t head_dim) {
  std::size_t d = 0;
  for (; d + 32 <= head_dim; d += 32) {
    float32x4_t acc[8];
    for (auto& a : acc) {
      a = vdupq_n_f32(0.0F);
    }
    for (std::size_t j = 0; j < count; ++j) {
      const float32x4_t pj = vdupq_n_f32(p[j]);
      const float* v = v_cache + j * v_stride + d;
      for (std::size_t g = 0; g < 8; ++g) {
        acc[g] = vfmaq_f32(acc[g], pj, vld1q_f32(v + 4 * g));
      }
    }
    for (std::size_t g = 0; g < 8; ++g) {
      vst1q_f32(o + d + 4 * g, acc[g]);
    }
  }
  for (; d + 4 <= head_dim; d += 4) {
    float32x4_t acc = vdupq_n_f32(0.0F);
    for (std::size_t j = 0; j < count; ++j) {
      acc = vfmaq_f32(acc, vdupq_n_f32(p[j]), vld1q_f32(v_cache + j * v_stride + d));
    }
    vst1q_f32(o + d, acc);
  }
  for (; d < head_dim; ++d) {
    float acc = 0.0F;
    for (std::size_t j = 0; j < count; ++j) {
      acc += p[j] * v_cache[j * v_stride + d];
    }
    o[d] = acc;
  }
}

void attention_kv_heads(const float* q, const float* k_cache, const float* v_cache, float* out,
                        float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                        std::size_t n_head_kv, std::size_t head_dim, float scale,
                        std::size_t head_begin, std::size_t head_end) {
  AXON_DCHECK(n_head_kv > 0 && n_head % n_head_kv == 0, "attention: {} heads over {} kv heads",
              n_head, n_head_kv);
  AXON_DCHECK(head_begin <= head_end && head_end <= n_head, "attention: heads [{}, {}) of {}",
              head_begin, head_end, n_head);
  const std::size_t group = n_head / n_head_kv;
  const std::size_t total = n_past + n_q;
  const std::size_t kv_stride = n_head_kv * head_dim;
  const auto softmax = scalar_kernels().softmax;
  for (std::size_t h = head_begin; h < head_end; ++h) {
    const std::size_t kv_head = h / group;
    const float* k_head = k_cache + kv_head * head_dim;
    const float* v_head = v_cache + kv_head * head_dim;
    for (std::size_t i = 0; i < n_q; ++i) {
      const std::size_t pos = n_past + i;
      float* row = scores + i * total;
      const float* q_i = q + (i * n_head + h) * head_dim;
      for (std::size_t j = 0; j <= pos; ++j) {
        row[j] = scale * dot_q_k(q_i, k_head + j * kv_stride, head_dim);
      }
      softmax(row, pos + 1);
      weighted_values(row, v_head, kv_stride, pos + 1, out + (i * n_head + h) * head_dim, head_dim);
    }
  }
}

}  // namespace

void install_attention_f32(KernelTable& table) {
  table.attention_kv_heads = attention_kv_heads;
}

}  // namespace axon::kernels::neon
