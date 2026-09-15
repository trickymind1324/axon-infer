#include "core/check.h"

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

// One head at a time: fill the lower triangle of the score matrix, softmax each row over the
// positions it may see, then mix the values. Quadratic in seq and unapologetically so; this is
// the baseline the cached and blocked versions are measured against.
void attention(const float* q, const float* k, const float* v, float* out, float* scores,
               std::size_t seq, std::size_t n_head, std::size_t n_head_kv, std::size_t head_dim,
               float scale) {
  AXON_DCHECK(n_head_kv > 0 && n_head % n_head_kv == 0, "attention: {} heads over {} kv heads",
              n_head, n_head_kv);
  const std::size_t group = n_head / n_head_kv;
  for (std::size_t h = 0; h < n_head; ++h) {
    const std::size_t kv_head = h / group;
    for (std::size_t i = 0; i < seq; ++i) {
      float* row = scores + i * seq;
      const float* q_i = q + (i * n_head + h) * head_dim;
      for (std::size_t j = 0; j <= i; ++j) {
        row[j] = scale * dot(q_i, k + (j * n_head_kv + kv_head) * head_dim, head_dim);
      }
      softmax(row, i + 1);
      float* o = out + (i * n_head + h) * head_dim;
      for (std::size_t d = 0; d < head_dim; ++d) {
        o[d] = 0.0F;
      }
      for (std::size_t j = 0; j <= i; ++j) {
        const float* v_j = v + (j * n_head_kv + kv_head) * head_dim;
        for (std::size_t d = 0; d < head_dim; ++d) {
          o[d] += row[j] * v_j[d];
        }
      }
    }
  }
}

}  // namespace axon::kernels::ref
