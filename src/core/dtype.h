#pragma once

#include "core/expected.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace axon {

// Tensor element types. Values are the ggml type ids as written in GGUF files. Every id ggml
// has assigned is listed so any file can be inspected; only a few are runnable (see
// DTypeInfo::compute_supported).
enum class DType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  TQ1_0 = 34,
  TQ2_0 = 35,
};

struct DTypeInfo {
  DType type;
  std::string_view name;
  std::uint32_t block_size;  // elements per block
  std::uint32_t type_size;   // bytes per block
  bool compute_supported;    // the engine has kernels for it
};

// nullptr for ids ggml never assigned (or has removed).
const DTypeInfo* dtype_info(std::uint32_t ggml_id);
inline const DTypeInfo* dtype_info(DType type) {
  return dtype_info(static_cast<std::uint32_t>(type));
}
std::string_view dtype_name(DType type);  // "unknown(<id>)" for unlisted ids

constexpr bool is_quantized(DType type) {
  switch (type) {
  case DType::F32:
  case DType::F16:
  case DType::BF16:
  case DType::F64:
  case DType::I8:
  case DType::I16:
  case DType::I32:
  case DType::I64:
    return false;
  default:
    return true;
  }
}

// Bytes occupied by a row of n elements. Fails when n is not a multiple of the block size.
Expected<std::size_t> row_bytes(DType type, std::uint64_t n_elements);

// Exact ggml block layouts.
constexpr std::uint32_t kQK8_0 = 32;
constexpr std::uint32_t kQK4_0 = 32;

struct BlockQ8_0 {
  std::uint16_t d;         // fp16 scale
  std::int8_t qs[kQK8_0];  // value = qs[i] * d
};
static_assert(sizeof(BlockQ8_0) == 34);

struct BlockQ4_0 {
  std::uint16_t d;              // fp16 scale
  std::uint8_t qs[kQK4_0 / 2];  // element i is the low nibble of qs[i], element i+16 the high
                                // nibble; value = (nibble - 8) * d
};
static_assert(sizeof(BlockQ4_0) == 18);

// Scalar conversions, exact for every bit pattern (subnormals, inf, nan preserved).
float fp16_to_fp32(std::uint16_t bits);
std::uint16_t fp32_to_fp16(float value);  // round to nearest even
float bf16_to_fp32(std::uint16_t bits);
std::uint16_t fp32_to_bf16(float value);  // round to nearest even, nan kept quiet

}  // namespace axon
