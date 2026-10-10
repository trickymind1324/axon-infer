#include "core/check.h"
#include "core/dtype.h"

#include "kernels/ref/ref.h"

#include <cstddef>

namespace axon::kernels::ref {

namespace {

float dot(const float* a, const float* b, std::size_t n) {
  float acc = 0.0F;
  for (std::size_t i = 0; i < n; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

}  // namespace

// One head at a time: fill each query's score row over the positions it may see, softmax it,
// then mix the values in ascending order. Every reduction runs in the same order whether the
// keys arrive all at once or one position at a time, which is what makes cached decoding
// bit-identical to the full-sequence pass. Quadratic and unapologetically so: this is the
// baseline the blocked versions are measured against.
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
  for (std::size_t h = head_begin; h < head_end; ++h) {
    const std::size_t kv_head = h / group;
    for (std::size_t i = 0; i < n_q; ++i) {
      const std::size_t pos = n_past + i;
      float* row = scores + i * total;
      const float* q_i = q + (i * n_head + h) * head_dim;
      for (std::size_t j = 0; j <= pos; ++j) {
        row[j] = scale * dot(q_i, k_cache + (j * n_head_kv + kv_head) * head_dim, head_dim);
      }
      softmax(row, pos + 1);
      float* o = out + (i * n_head + h) * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        o[d] = 0.0F;
      }
      for (std::size_t j = 0; j <= pos; ++j) {
        const float* v_j = v_cache + (j * n_head_kv + kv_head) * head_dim;
        for (std::size_t d = 0; d < head_dim; ++d) {
          o[d] += row[j] * v_j[d];
        }
      }
    }
  }
}

void attention_kv(const float* q, const float* k_cache, const float* v_cache, float* out,
                  float* scores, std::size_t n_q, std::size_t n_past, std::size_t n_head,
                  std::size_t n_head_kv, std::size_t head_dim, float scale) {
  attention_kv_heads(q, k_cache, v_cache, out, scores, n_q, n_past, n_head, n_head_kv, head_dim,
                     scale, 0, n_head);
}

void attention(const float* q, const float* k, const float* v, float* out, float* scores,
               std::size_t seq, std::size_t n_head, std::size_t n_head_kv, std::size_t head_dim,
               float scale) {
  attention_kv(q, k, v, out, scores, seq, 0, n_head, n_head_kv, head_dim, scale);
}

// The fp16 cache variant: identical loop structure, with every key and value widened on the
// way in. The widening is exact, so against the F32 cache the only difference is the rounding
// that happened when the cache row was written.
void attention_kv_heads_f16(const float* q, const std::uint16_t* k_cache,
                            const std::uint16_t* v_cache, float* out, float* scores,
                            std::size_t n_q, std::size_t n_past, std::size_t n_head,
                            std::size_t n_head_kv, std::size_t head_dim, float scale,
                            std::size_t head_begin, std::size_t head_end) {
  AXON_DCHECK(n_head_kv > 0 && n_head % n_head_kv == 0, "attention: {} heads over {} kv heads",
              n_head, n_head_kv);
  const std::size_t group = n_head / n_head_kv;
  const std::size_t total = n_past + n_q;
  for (std::size_t h = head_begin; h < head_end; ++h) {
    const std::size_t kv_head = h / group;
    for (std::size_t i = 0; i < n_q; ++i) {
      const std::size_t pos = n_past + i;
      float* row = scores + i * total;
      const float* q_i = q + (i * n_head + h) * head_dim;
      for (std::size_t j = 0; j <= pos; ++j) {
        const std::uint16_t* k_j = k_cache + (j * n_head_kv + kv_head) * head_dim;
        float acc = 0.0F;
        for (std::size_t d = 0; d < head_dim; ++d) {
          acc += q_i[d] * fp16_to_fp32(k_j[d]);
        }
        row[j] = scale * acc;
      }
      softmax(row, pos + 1);
      float* o = out + (i * n_head + h) * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        o[d] = 0.0F;
      }
      for (std::size_t j = 0; j <= pos; ++j) {
        const std::uint16_t* v_j = v_cache + (j * n_head_kv + kv_head) * head_dim;
        for (std::size_t d = 0; d < head_dim; ++d) {
          o[d] += row[j] * fp16_to_fp32(v_j[d]);
        }
      }
    }
  }
}

}  // namespace axon::kernels::ref
