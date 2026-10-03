#include "core/dtype.h"

#include "core/check.h"

#include <fmt/format.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <string>

namespace axon {

namespace {

constexpr DTypeInfo kTypes[] = {
    {.type = DType::F32, .name = "F32", .block_size = 1, .type_size = 4, .compute_supported = true},
    {.type = DType::F16, .name = "F16", .block_size = 1, .type_size = 2, .compute_supported = true},
    {.type = DType::Q4_0,
     .name = "Q4_0",
     .block_size = 32,
     .type_size = 18,
     .compute_supported = true},
    {.type = DType::Q4_1,
     .name = "Q4_1",
     .block_size = 32,
     .type_size = 20,
     .compute_supported = true},
    {.type = DType::Q5_0,
     .name = "Q5_0",
     .block_size = 32,
     .type_size = 22,
     .compute_supported = false},
    {.type = DType::Q5_1,
     .name = "Q5_1",
     .block_size = 32,
     .type_size = 24,
     .compute_supported = false},
    {.type = DType::Q8_0,
     .name = "Q8_0",
     .block_size = 32,
     .type_size = 34,
     .compute_supported = true},
    {.type = DType::Q8_1,
     .name = "Q8_1",
     .block_size = 32,
     .type_size = 36,
     .compute_supported = false},
    {.type = DType::Q2_K,
     .name = "Q2_K",
     .block_size = 256,
     .type_size = 84,
     .compute_supported = false},
    {.type = DType::Q3_K,
     .name = "Q3_K",
     .block_size = 256,
     .type_size = 110,
     .compute_supported = false},
    {.type = DType::Q4_K,
     .name = "Q4_K",
     .block_size = 256,
     .type_size = 144,
     .compute_supported = false},
    {.type = DType::Q5_K,
     .name = "Q5_K",
     .block_size = 256,
     .type_size = 176,
     .compute_supported = false},
    {.type = DType::Q6_K,
     .name = "Q6_K",
     .block_size = 256,
     .type_size = 210,
     .compute_supported = true},
    {.type = DType::Q8_K,
     .name = "Q8_K",
     .block_size = 256,
     .type_size = 292,
     .compute_supported = false},
    {.type = DType::IQ2_XXS,
     .name = "IQ2_XXS",
     .block_size = 256,
     .type_size = 66,
     .compute_supported = false},
    {.type = DType::IQ2_XS,
     .name = "IQ2_XS",
     .block_size = 256,
     .type_size = 74,
     .compute_supported = false},
    {.type = DType::IQ3_XXS,
     .name = "IQ3_XXS",
     .block_size = 256,
     .type_size = 98,
     .compute_supported = false},
    {.type = DType::IQ1_S,
     .name = "IQ1_S",
     .block_size = 256,
     .type_size = 50,
     .compute_supported = false},
    {.type = DType::IQ4_NL,
     .name = "IQ4_NL",
     .block_size = 32,
     .type_size = 18,
     .compute_supported = false},
    {.type = DType::IQ3_S,
     .name = "IQ3_S",
     .block_size = 256,
     .type_size = 110,
     .compute_supported = false},
    {.type = DType::IQ2_S,
     .name = "IQ2_S",
     .block_size = 256,
     .type_size = 82,
     .compute_supported = false},
    {.type = DType::IQ4_XS,
     .name = "IQ4_XS",
     .block_size = 256,
     .type_size = 136,
     .compute_supported = false},
    {.type = DType::I8, .name = "I8", .block_size = 1, .type_size = 1, .compute_supported = false},
    {.type = DType::I16,
     .name = "I16",
     .block_size = 1,
     .type_size = 2,
     .compute_supported = false},
    {.type = DType::I32,
     .name = "I32",
     .block_size = 1,
     .type_size = 4,
     .compute_supported = false},
    {.type = DType::I64,
     .name = "I64",
     .block_size = 1,
     .type_size = 8,
     .compute_supported = false},
    {.type = DType::F64,
     .name = "F64",
     .block_size = 1,
     .type_size = 8,
     .compute_supported = false},
    {.type = DType::IQ1_M,
     .name = "IQ1_M",
     .block_size = 256,
     .type_size = 56,
     .compute_supported = false},
    {.type = DType::BF16,
     .name = "BF16",
     .block_size = 1,
     .type_size = 2,
     .compute_supported = true},
    {.type = DType::TQ1_0,
     .name = "TQ1_0",
     .block_size = 256,
     .type_size = 54,
     .compute_supported = false},
    {.type = DType::TQ2_0,
     .name = "TQ2_0",
     .block_size = 256,
     .type_size = 66,
     .compute_supported = false},
};

}  // namespace

const DTypeInfo* dtype_info(std::uint32_t ggml_id) {
  for (const DTypeInfo& info : kTypes) {
    if (static_cast<std::uint32_t>(info.type) == ggml_id) {
      return &info;
    }
  }
  return nullptr;
}

std::string_view dtype_name(DType type) {
  const DTypeInfo* info = dtype_info(type);
  if (info != nullptr) {
    return info->name;
  }
  // Rare path; a small static ring of names avoids returning a dangling view.
  thread_local std::string unknown;
  unknown = fmt::format("unknown({})", static_cast<std::uint32_t>(type));
  return unknown;
}

Expected<std::size_t> row_bytes(DType type, std::uint64_t n_elements) {
  const DTypeInfo* info = dtype_info(type);
  if (info == nullptr) {
    return make_error(ErrorCode::Unsupported,
                      fmt::format("unknown tensor type id {}", static_cast<std::uint32_t>(type)));
  }
  if (n_elements % info->block_size != 0) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{} row length: expected a multiple of {}, found {}", info->name,
                                  info->block_size, n_elements));
  }
  return static_cast<std::size_t>(n_elements / info->block_size * info->type_size);
}

float fp16_to_fp32(std::uint16_t bits) {
  const std::uint32_t sign = (bits & 0x8000U) << 16;
  const std::uint32_t exponent = (bits >> 10) & 0x1FU;
  const std::uint32_t mantissa = bits & 0x3FFU;
  std::uint32_t out = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      out = sign;  // signed zero
    } else {
      // Subnormal: renormalise.
      std::uint32_t m = mantissa;
      std::uint32_t e = 127 - 15 + 1;
      while ((m & 0x400U) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFU;
      out = sign | (e << 23) | (m << 13);
    }
  } else if (exponent == 0x1F) {
    out = sign | 0x7F800000U | (mantissa << 13);  // inf or nan (payload kept)
  } else {
    out = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(out);
}

std::uint16_t fp32_to_fp16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent == 0xFF) {
    // inf or nan; keep nan non-zero.
    const auto payload = static_cast<std::uint16_t>(mantissa >> 13);
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (mantissa != 0 && payload == 0 ? 1U : payload));
  }

  const int unbiased = static_cast<int>(exponent) - 127;
  if (unbiased > 15) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);  // overflow to inf
  }
  if (unbiased >= -14) {
    // Normal range: round mantissa from 23 to 10 bits, nearest even.
    auto half_exp = static_cast<std::uint32_t>(unbiased + 15);
    std::uint32_t half_man = mantissa >> 13;
    const std::uint32_t rest = mantissa & 0x1FFFU;
    if (rest > 0x1000U || (rest == 0x1000U && (half_man & 1U) != 0)) {
      ++half_man;
      if (half_man == 0x400U) {  // mantissa overflow carries into the exponent
        half_man = 0;
        ++half_exp;
        if (half_exp == 0x1F) {
          return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
      }
    }
    return static_cast<std::uint16_t>(sign | (half_exp << 10) | half_man);
  }
  if (unbiased >= -25) {
    // Subnormal half: shift the implicit-1 mantissa right, rounding nearest even.
    const std::uint32_t full = mantissa | 0x800000U;
    const int shift = -14 - unbiased + 13;  // 14..24
    std::uint32_t half_man = full >> shift;
    const std::uint32_t rest = full & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (rest > halfway || (rest == halfway && (half_man & 1U) != 0)) {
      ++half_man;  // may become 0x400 == smallest normal, which is the right encoding
    }
    return static_cast<std::uint16_t>(sign | half_man);
  }
  return sign;  // underflow to signed zero
}

float bf16_to_fp32(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

std::uint16_t fp32_to_bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  if (((bits >> 23) & 0xFFU) == 0xFFU && (bits & 0x7FFFFFU) != 0) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x40U);  // quiet nan
  }
  const std::uint32_t lsb = (bits >> 16) & 1U;
  const std::uint32_t rounded = bits + 0x7FFFU + lsb;  // nearest even
  return static_cast<std::uint16_t>(rounded >> 16);
}

void dequantize_block_q8_0(const BlockQ8_0& block, float* out) {
  const float d = fp16_to_fp32(block.d);
  for (std::uint32_t i = 0; i < kQK8_0; ++i) {
    out[i] = static_cast<float>(block.qs[i]) * d;
  }
}

void dequantize_block_q4_0(const BlockQ4_0& block, float* out) {
  const float d = fp16_to_fp32(block.d);
  for (std::uint32_t i = 0; i < kQK4_0 / 2; ++i) {
    out[i] = static_cast<float>(static_cast<int>(block.qs[i] & 0x0F) - 8) * d;
    out[i + kQK4_0 / 2] = static_cast<float>(static_cast<int>(block.qs[i] >> 4) - 8) * d;
  }
}

void dequantize_block_q4_1(const BlockQ4_1& block, float* out) {
  const float d = fp16_to_fp32(block.d);
  const float m = fp16_to_fp32(block.m);
  for (std::uint32_t i = 0; i < kQK4_1 / 2; ++i) {
    out[i] = static_cast<float>(block.qs[i] & 0x0F) * d + m;
    out[i + kQK4_1 / 2] = static_cast<float>(block.qs[i] >> 4) * d + m;
  }
}

// ggml's dequantize_row_q6_K, including its evaluation order (d * scale) * q.
void dequantize_block_q6_k(const BlockQ6_K& block, float* out) {
  const float d = fp16_to_fp32(block.d);
  const std::uint8_t* ql = block.ql;
  const std::uint8_t* qh = block.qh;
  const std::int8_t* sc = block.scales;
  float* y = out;
  for (std::uint32_t n = 0; n < kQK_K; n += 128) {
    for (std::uint32_t l = 0; l < 32; ++l) {
      const std::uint32_t is = l / 16;
      const int q1 = static_cast<int>((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) - 32;
      const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l] = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
      y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
      y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
      y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

void quantize_row_q8_0(const float* x, BlockQ8_0* out, std::size_t n) {
  AXON_DCHECK(n % kQK8_0 == 0, "quantize_row_q8_0: {} is not a multiple of {}", n, kQK8_0);
  for (std::size_t b = 0; b < n / kQK8_0; ++b) {
    const float* block = x + b * kQK8_0;
    float amax = 0.0F;
    for (std::uint32_t i = 0; i < kQK8_0; ++i) {
      const float a = std::fabs(block[i]);
      amax = a > amax ? a : amax;
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    out[b].d = fp32_to_fp16(d);
    for (std::uint32_t i = 0; i < kQK8_0; ++i) {
      out[b].qs[i] = static_cast<std::int8_t>(std::roundf(block[i] * id));
    }
  }
}

}  // namespace axon
