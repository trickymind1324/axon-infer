#pragma once

// Scalar reference implementations. Plain loops, no intrinsics, written to be read.

#include "kernels/kernels.h"

#include <cstddef>
#include <cstdint>

namespace axon::kernels::ref {

void rmsnorm(const float* x, const float* weight, float* y, std::size_t n, float eps);
void softmax(float* x, std::size_t n);
void silu_mul(const float* gate, const float* up, float* out, std::size_t n);

}  // namespace axon::kernels::ref
