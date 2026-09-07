#include "core/dtype.h"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

TEST_CASE("dtype table lookups", "[core][dtype]") {
  const auto* q4 = axon::dtype_info(2);
  REQUIRE(q4 != nullptr);
  CHECK(q4->type == axon::DType::Q4_0);
  CHECK(q4->name == "Q4_0");
  CHECK(q4->block_size == 32);
  CHECK(q4->type_size == 18);
  CHECK(q4->compute_supported);

  const auto* q4k = axon::dtype_info(axon::DType::Q4_K);
  REQUIRE(q4k != nullptr);
  CHECK(q4k->block_size == 256);
  CHECK(q4k->type_size == 144);
  CHECK_FALSE(q4k->compute_supported);

  CHECK(axon::dtype_info(4) == nullptr);  // removed id
  CHECK(axon::dtype_info(5) == nullptr);
  CHECK(axon::dtype_info(31) == nullptr);
  CHECK(axon::dtype_info(999) == nullptr);
  CHECK(axon::dtype_name(axon::DType::BF16) == "BF16");
  CHECK(axon::dtype_name(static_cast<axon::DType>(999)) == "unknown(999)");

  CHECK(axon::is_quantized(axon::DType::Q8_0));
  CHECK_FALSE(axon::is_quantized(axon::DType::F16));
  CHECK_FALSE(axon::is_quantized(axon::DType::I32));
}

TEST_CASE("row_bytes for compute types", "[core][dtype]") {
  CHECK(axon::row_bytes(axon::DType::F32, 64).value() == 256);
  CHECK(axon::row_bytes(axon::DType::F16, 64).value() == 128);
  CHECK(axon::row_bytes(axon::DType::BF16, 64).value() == 128);
  CHECK(axon::row_bytes(axon::DType::Q8_0, 64).value() == 68);
  CHECK(axon::row_bytes(axon::DType::Q4_0, 64).value() == 36);
  CHECK(axon::row_bytes(axon::DType::Q4_K, 512).value() == 288);

  const auto bad = axon::row_bytes(axon::DType::Q4_0, 33);
  REQUIRE_FALSE(bad);
  CHECK(bad.error().code == axon::ErrorCode::InvalidArgument);
  CHECK(bad.error().message == "Q4_0 row length: expected a multiple of 32, found 33");

  const auto unknown = axon::row_bytes(static_cast<axon::DType>(5), 32);
  REQUIRE_FALSE(unknown);
  CHECK(unknown.error().code == axon::ErrorCode::Unsupported);
}

TEST_CASE("fp16 conversion is exact for every bit pattern", "[core][dtype]") {
  for (std::uint32_t bits = 0; bits < 0x10000U; ++bits) {
    const auto half = static_cast<std::uint16_t>(bits);
    const float f = axon::fp16_to_fp32(half);
    const std::uint16_t back = axon::fp32_to_fp16(f);
    if (std::isnan(f)) {
      CHECK(std::isnan(axon::fp16_to_fp32(back)));
      CHECK((back & 0x8000U) == (half & 0x8000U));
    } else {
      REQUIRE(back == half);
    }
  }
}

TEST_CASE("fp16 known values", "[core][dtype]") {
  CHECK(axon::fp16_to_fp32(0x3C00) == 1.0F);
  CHECK(axon::fp16_to_fp32(0xC000) == -2.0F);
  CHECK(axon::fp16_to_fp32(0x7BFF) == 65504.0F);
  CHECK(axon::fp16_to_fp32(0x0001) == std::ldexp(1.0F, -24));  // smallest subnormal
  CHECK(axon::fp16_to_fp32(0x0400) == std::ldexp(1.0F, -14));  // smallest normal
  CHECK(std::isinf(axon::fp16_to_fp32(0x7C00)));
  CHECK(std::signbit(axon::fp16_to_fp32(0x8000)));

  CHECK(axon::fp32_to_fp16(0.1F) == 0x2E66);
  CHECK(axon::fp32_to_fp16(1e5F) == 0x7C00);  // overflow to inf
  CHECK(axon::fp32_to_fp16(-1e5F) == 0xFC00);
  CHECK(axon::fp32_to_fp16(1e-9F) == 0x0000);  // underflow to zero
  CHECK(axon::fp32_to_fp16(-1e-9F) == 0x8000);
  CHECK(axon::fp32_to_fp16(std::ldexp(1.0F, -25)) ==
        0x0000);  // exactly half of min subnormal: ties to even
  CHECK(axon::fp32_to_fp16(std::ldexp(1.5F, -25)) == 0x0001);
  CHECK(axon::fp32_to_fp16(2049.0F) == 0x6800);   // tie rounds to even (2048)
  CHECK(axon::fp32_to_fp16(2051.0F) == 0x6802);   // tie rounds to even (2052)
  CHECK(axon::fp32_to_fp16(65520.0F) == 0x7C00);  // rounds up past max into inf
  CHECK((axon::fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()) & 0x7C00U) == 0x7C00U);
  CHECK((axon::fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()) & 0x03FFU) != 0);
}

TEST_CASE("bf16 conversion", "[core][dtype]") {
  CHECK(axon::bf16_to_fp32(0x3F80) == 1.0F);
  CHECK(axon::bf16_to_fp32(0xC040) == -3.0F);
  CHECK(axon::fp32_to_bf16(1.0F) == 0x3F80);
  CHECK(axon::fp32_to_bf16(std::bit_cast<float>(0x3F808000U)) == 0x3F80);  // tie to even (down)
  CHECK(axon::fp32_to_bf16(std::bit_cast<float>(0x3F818000U)) == 0x3F82);  // tie to even (up)
  CHECK(axon::fp32_to_bf16(std::bit_cast<float>(0x3F808001U)) == 0x3F81);  // above tie rounds up
  CHECK(
      std::isnan(axon::bf16_to_fp32(axon::fp32_to_bf16(std::numeric_limits<float>::quiet_NaN()))));
  CHECK(std::isinf(axon::bf16_to_fp32(axon::fp32_to_bf16(std::numeric_limits<float>::infinity()))));
  // Every bf16 value is representable exactly in fp32 and round-trips.
  for (std::uint32_t bits = 0; bits < 0x10000U; ++bits) {
    const auto b = static_cast<std::uint16_t>(bits);
    const float f = axon::bf16_to_fp32(b);
    if (!std::isnan(f)) {
      REQUIRE(axon::fp32_to_bf16(f) == b);
    }
  }
}
