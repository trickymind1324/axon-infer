#include "core/check.h"

#include "kernels/ref/ref.h"

#include <cmath>
#include <cstddef>

namespace axon::kernels::ref {

void softmax(float* x, std::size_t n) {
  AXON_DCHECK(n > 0);
  float max = x[0];
  for (std::size_t i = 1; i < n; ++i) {
    max = x[i] > max ? x[i] : max;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - max);
    sum += static_cast<double>(x[i]);
  }
  const auto inv_sum = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] *= inv_sum;
  }
}

}  // namespace axon::kernels::ref
