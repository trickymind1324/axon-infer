#include "core/check.h"

#include "kernels/ref/ref.h"

#include <cmath>
#include <cstddef>

namespace axon::kernels::ref {

void rmsnorm(const float* x, const float* weight, float* y, std::size_t n, float eps) {
  AXON_DCHECK(n > 0);
  double sum_squares = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum_squares += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const auto inv_rms = static_cast<float>(
      1.0 / std::sqrt(sum_squares / static_cast<double>(n) + static_cast<double>(eps)));
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = x[i] * inv_rms * weight[i];
  }
}

}  // namespace axon::kernels::ref
