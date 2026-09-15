#include "core/check.h"

#include "kernels/ref/ref.h"

#include <cmath>
#include <cstddef>

namespace axon::kernels::ref {

// inv_freq[i] = theta^(-2i / rotary_dim), divided by freq_factors[i] when given. Angles and
// their sin/cos are computed in double so that positions in the thousands stay accurate; the
// rotation itself runs in float.
void rope(float* x, std::size_t n_heads, std::size_t pos, const RopeParams& params) {
  AXON_DCHECK(params.rotary_dim % 2 == 0 && params.rotary_dim <= params.head_dim,
              "rope: rotary_dim {} head_dim {}", params.rotary_dim, params.head_dim);
  const std::size_t half = params.rotary_dim / 2;
  const auto theta = static_cast<double>(params.theta);
  const auto rotary_dim = static_cast<double>(params.rotary_dim);
  for (std::size_t i = 0; i < half; ++i) {
    double inv_freq = std::pow(theta, -static_cast<double>(2 * i) / rotary_dim);
    if (params.freq_factors != nullptr) {
      inv_freq /= static_cast<double>(params.freq_factors[i]);
    }
    const double angle = static_cast<double>(pos) * inv_freq;
    const auto c = static_cast<float>(std::cos(angle));
    const auto s = static_cast<float>(std::sin(angle));
    const std::size_t a = params.style == RopeStyle::Normal ? 2 * i : i;
    const std::size_t b = params.style == RopeStyle::Normal ? 2 * i + 1 : i + half;
    for (std::size_t h = 0; h < n_heads; ++h) {
      float* head = x + h * params.head_dim;
      const float x0 = head[a];
      const float x1 = head[b];
      head[a] = x0 * c - x1 * s;
      head[b] = x0 * s + x1 * c;
    }
  }
}

}  // namespace axon::kernels::ref
