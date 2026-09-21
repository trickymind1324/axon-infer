#include "core/dtype.h"

#include "fixture_tensor.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

template <class Block> std::vector<Block> blocks_of(const std::string& name) {
  const auto bytes = axon::test::read_fixture("quant/" + name);
  REQUIRE(bytes.size() % sizeof(Block) == 0);
  std::vector<Block> out(bytes.size() / sizeof(Block));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

}  // namespace

TEST_CASE("dequantizing Q8_0 and Q4_0 blocks matches the gguf package", "[core][quant]") {
  const auto q8 = blocks_of<axon::BlockQ8_0>("row.q8_0.bin");
  const auto expected8 = axon::test::load_axt("row.q8_0.dequant", "quant").as_f32();
  REQUIRE(q8.size() * 32 == expected8.size());
  std::vector<float> got8(expected8.size());
  for (std::size_t b = 0; b < q8.size(); ++b) {
    axon::dequantize_block_q8_0(q8[b], got8.data() + b * 32);
  }
  CHECK(got8 == expected8);

  const auto q4 = blocks_of<axon::BlockQ4_0>("row.q4_0.bin");
  const auto expected4 = axon::test::load_axt("row.q4_0.dequant", "quant").as_f32();
  REQUIRE(q4.size() * 32 == expected4.size());
  std::vector<float> got4(expected4.size());
  for (std::size_t b = 0; b < q4.size(); ++b) {
    axon::dequantize_block_q4_0(q4[b], got4.data() + b * 32);
  }
  CHECK(got4 == expected4);
}

TEST_CASE("quantizing a row to Q8_0 matches the gguf package bit for bit", "[core][quant]") {
  for (const char* name : {"row", "act"}) {
    const std::string base = name;
    const auto x = axon::test::load_axt(base + ".f32", "quant").as_f32();
    const auto expected = blocks_of<axon::BlockQ8_0>(base + ".q8_0.bin");
    REQUIRE(expected.size() == x.size() / 32);
    std::vector<axon::BlockQ8_0> got(expected.size());
    axon::quantize_row_q8_0(x.data(), got.data(), x.size());
    INFO(name);
    CHECK(std::memcmp(got.data(), expected.data(), got.size() * sizeof(axon::BlockQ8_0)) == 0);
  }
}

// The scale is stored as fp16, so a dequantized value can miss the original by half a quantum
// plus the fp16 rounding of the scale times the integer, which is at most 127 * 2^-11 quanta.
TEST_CASE("Q8_0 quantization round trip stays within a rounded quantum", "[core][quant]") {
  std::vector<float> x(64);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(static_cast<int>(i) - 31) * 0.37F;
  }
  std::vector<axon::BlockQ8_0> q(2);
  axon::quantize_row_q8_0(x.data(), q.data(), x.size());
  std::vector<float> back(64);
  axon::dequantize_block_q8_0(q[0], back.data());
  axon::dequantize_block_q8_0(q[1], back.data() + 32);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float d = axon::fp16_to_fp32(q[i / 32].d);
    INFO("element " << i);
    CHECK(std::abs(back[i] - x[i]) <= d * 0.57F + 1e-6F);
  }
  // The largest magnitude maps to exactly 127 or -127.
  bool saw_extreme = false;
  for (const auto& block : q) {
    for (const auto v : block.qs) {
      saw_extreme = saw_extreme || v == 127 || v == -127;
    }
  }
  CHECK(saw_extreme);
}
