#include "core/check.h"
#include "core/dtype.h"

#include "kernels/ref/ref.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace axon::kernels::ref {

void embedding_row(DType type, const void* table, std::size_t n, std::int64_t index, float* out) {
  AXON_DCHECK(index >= 0, "embedding row {}", index);
  const auto row = static_cast<std::size_t>(index);
  switch (type) {
  case DType::F32:
    std::memcpy(out, static_cast<const float*>(table) + row * n, n * sizeof(float));
    return;
  case DType::F16: {
    const std::uint16_t* src = static_cast<const std::uint16_t*>(table) + row * n;
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = fp16_to_fp32(src[i]);
    }
    return;
  }
  case DType::BF16: {
    const std::uint16_t* src = static_cast<const std::uint16_t*>(table) + row * n;
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = bf16_to_fp32(src[i]);
    }
    return;
  }
  case DType::Q8_0: {
    const BlockQ8_0* src = static_cast<const BlockQ8_0*>(table) + row * (n / kQK8_0);
    for (std::size_t b = 0; b < n / kQK8_0; ++b) {
      dequantize_block_q8_0(src[b], out + b * kQK8_0);
    }
    return;
  }
  case DType::Q4_0: {
    const BlockQ4_0* src = static_cast<const BlockQ4_0*>(table) + row * (n / kQK4_0);
    for (std::size_t b = 0; b < n / kQK4_0; ++b) {
      dequantize_block_q4_0(src[b], out + b * kQK4_0);
    }
    return;
  }
  default:
    AXON_CHECK(false, "embedding_row: unsupported dtype {}", dtype_name(type));
  }
}

}  // namespace axon::kernels::ref
