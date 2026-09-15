#include "kernels/ref/ref.h"

#include <cmath>
#include <cstddef>

namespace axon::kernels::ref {

void silu_mul(const float* gate, const float* up, float* out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float g = gate[i];
    out[i] = g / (1.0F + std::exp(-g)) * up[i];
  }
}

}  // namespace axon::kernels::ref
