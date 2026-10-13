#include "core/dtype.h"

#include "kernels/ref/ref.h"

#include <cstddef>
#include <cstdint>

namespace axon::kernels::ref {

namespace {

// One dot product per output row, accumulated left to right in float. This order is the
// reference every other matmul is compared against.
template <class Weight, class Convert>
void gemv(const Weight* w, const float* x, float* y, std::size_t n_out, std::size_t n_in,
          Convert convert) {
  for (std::size_t o = 0; o < n_out; ++o) {
    const Weight* row = w + o * n_in;
    float acc = 0.0F;
    for (std::size_t i = 0; i < n_in; ++i) {
      acc += convert(row[i]) * x[i];
    }
    y[o] = acc;
  }
}

template <class Weight, class Convert>
void gemm_rows(const Weight* w, const float* x, float* y, std::size_t n_rows, std::size_t n_out,
               std::size_t n_in, std::size_t y_stride, Convert convert) {
  for (std::size_t r = 0; r < n_rows; ++r) {
    gemv(w, x + r * n_in, y + r * y_stride, n_out, n_in, convert);
  }
}

float identity(float v) {
  return v;
}

}  // namespace

void matmul_f32(const float* w, const float* x, float* y, std::size_t n_out, std::size_t n_in) {
  gemv(w, x, y, n_out, n_in, identity);
}

void matmul_f16(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                std::size_t n_in) {
  gemv(w, x, y, n_out, n_in, fp16_to_fp32);
}

void matmul_bf16(const std::uint16_t* w, const float* x, float* y, std::size_t n_out,
                 std::size_t n_in) {
  gemv(w, x, y, n_out, n_in, bf16_to_fp32);
}

void matmul_f32_rows(const float* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  gemm_rows(w, x, y, n_rows, n_out, n_in, y_stride, identity);
}

void matmul_f16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                     std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  gemm_rows(w, x, y, n_rows, n_out, n_in, y_stride, fp16_to_fp32);
}

void matmul_bf16_rows(const std::uint16_t* w, const float* x, float* y, std::size_t n_rows,
                      std::size_t n_out, std::size_t n_in, std::size_t y_stride) {
  gemm_rows(w, x, y, n_rows, n_out, n_in, y_stride, bf16_to_fp32);
}

}  // namespace axon::kernels::ref
