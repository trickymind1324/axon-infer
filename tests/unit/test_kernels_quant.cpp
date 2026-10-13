#include "core/dtype.h"

#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

const axon::kernels::KernelTable& k() {
  return axon::kernels::kernels();
}

template <class Block> std::vector<Block> blocks_of(const std::string& name) {
  const auto bytes = axon::test::read_fixture("quant/" + name);
  REQUIRE(bytes.size() % sizeof(Block) == 0);
  std::vector<Block> out(bytes.size() / sizeof(Block));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

constexpr std::size_t kOut = 7;
constexpr std::size_t kIn = 96;

}  // namespace

TEST_CASE("quantized matmul with float activations matches dequantized weights",
          "[kernels][quant]") {
  const auto x = axon::test::load_axt("matmul_q8_0.x", "quant").as_f32();
  const auto expected8 = axon::test::load_axt("matmul_q8_0.y", "quant").as_f32();
  const auto w8 = blocks_of<axon::BlockQ8_0>("matmul_q8_0.w.bin");
  REQUIRE(w8.size() == kOut * kIn / 32);
  std::vector<float> y(kOut);
  k().matmul_q8_0(w8.data(), x.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, expected8) < 1e-4F);

  const auto x4 = axon::test::load_axt("matmul_q4_0.x", "quant").as_f32();
  const auto expected4 = axon::test::load_axt("matmul_q4_0.y", "quant").as_f32();
  const auto w4 = blocks_of<axon::BlockQ4_0>("matmul_q4_0.w.bin");
  k().matmul_q4_0(w4.data(), x4.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, expected4) < 1e-4F);
}

TEST_CASE("quantized matmul with Q8_0 activations matches the integer reference",
          "[kernels][quant]") {
  const auto expected8 = axon::test::load_axt("matmul_q8_0.yq", "quant").as_f32();
  const auto w8 = blocks_of<axon::BlockQ8_0>("matmul_q8_0.w.bin");
  const auto xq8 = blocks_of<axon::BlockQ8_0>("matmul_q8_0.xq.bin");
  REQUIRE(xq8.size() == kIn / 32);
  std::vector<float> y(kOut);
  k().matmul_q8_0_q8(w8.data(), xq8.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, expected8) < 1e-5F);

  const auto expected4 = axon::test::load_axt("matmul_q4_0.yq", "quant").as_f32();
  const auto w4 = blocks_of<axon::BlockQ4_0>("matmul_q4_0.w.bin");
  const auto xq4 = blocks_of<axon::BlockQ8_0>("matmul_q4_0.xq.bin");
  k().matmul_q4_0_q8(w4.data(), xq4.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, expected4) < 1e-5F);

  // Quantizing the activation through the table and then multiplying gives the same result.
  const auto x = axon::test::load_axt("matmul_q8_0.x", "quant").as_f32();
  std::vector<axon::BlockQ8_0> mine(kIn / 32);
  k().quantize_row_q8_0(x.data(), mine.data(), kIn);
  CHECK(std::memcmp(mine.data(), xq8.data(), mine.size() * sizeof(axon::BlockQ8_0)) == 0);
}

TEST_CASE("Q4_1 matmul matches on both activation paths", "[kernels][quant]") {
  const auto w = blocks_of<axon::BlockQ4_1>("matmul_q4_1.w.bin");
  const auto x = axon::test::load_axt("matmul_q4_1.x", "quant").as_f32();
  std::vector<float> y(kOut);
  k().matmul_q4_1(w.data(), x.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, axon::test::load_axt("matmul_q4_1.y", "quant").as_f32()) <
        1e-4F);
  const auto xq = blocks_of<axon::BlockQ8_0>("matmul_q4_1.xq.bin");
  k().matmul_q4_1_q8(w.data(), xq.data(), y.data(), kOut, kIn);
  CHECK(axon::test::max_abs_diff(y, axon::test::load_axt("matmul_q4_1.yq", "quant").as_f32()) <
        1e-5F);
}

TEST_CASE("Q6_K matmul dequantizes super-blocks", "[kernels][quant]") {
  const auto w = blocks_of<axon::BlockQ6_K>("matmul_q6_k.w.bin");
  REQUIRE(w.size() == 5);
  const auto x = axon::test::load_axt("matmul_q6_k.x", "quant").as_f32();
  std::vector<float> y(5);
  k().matmul_q6_k(w.data(), x.data(), y.data(), 5, 256);
  CHECK(axon::test::max_abs_diff(y, axon::test::load_axt("matmul_q6_k.y", "quant").as_f32()) <
        1e-3F);
  const auto xs = axon::test::load_axt("matmul_q6_k.xs", "quant").as_f32();
  std::vector<float> ys(std::size_t{3} * 5);
  k().matmul_q6_k_rows(w.data(), xs.data(), ys.data(), 3, 5, 256, 5);
  CHECK(axon::test::max_abs_diff(ys, axon::test::load_axt("matmul_q6_k.ys", "quant").as_f32()) <
        1e-3F);
}

TEST_CASE("quantized row matmuls match for both activation paths", "[kernels][quant]") {
  for (const char* name : {"matmul_q8_0", "matmul_q4_0", "matmul_q4_1"}) {
    const std::string base = name;
    const auto xs = axon::test::load_axt(base + ".xs", "quant").as_f32();
    const auto ys = axon::test::load_axt(base + ".ys", "quant").as_f32();
    const auto ysq = axon::test::load_axt(base + ".ysq", "quant").as_f32();
    const std::size_t rows = xs.size() / kIn;
    std::vector<float> y(rows * kOut);
    std::vector<axon::BlockQ8_0> xq(rows * kIn / 32);
    for (std::size_t r = 0; r < rows; ++r) {
      k().quantize_row_q8_0(xs.data() + r * kIn, xq.data() + r * (kIn / 32), kIn);
    }
    INFO(name);
    if (base == "matmul_q8_0") {
      const auto w = blocks_of<axon::BlockQ8_0>(base + ".w.bin");
      k().matmul_q8_0_rows(w.data(), xs.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ys) < 1e-4F);
      k().matmul_q8_0_q8_rows(w.data(), xq.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ysq) < 1e-5F);
    } else if (base == "matmul_q4_0") {
      const auto w = blocks_of<axon::BlockQ4_0>(base + ".w.bin");
      k().matmul_q4_0_rows(w.data(), xs.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ys) < 1e-4F);
      k().matmul_q4_0_q8_rows(w.data(), xq.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ysq) < 1e-5F);
    } else {
      const auto w = blocks_of<axon::BlockQ4_1>(base + ".w.bin");
      k().matmul_q4_1_rows(w.data(), xs.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ys) < 1e-4F);
      k().matmul_q4_1_q8_rows(w.data(), xq.data(), y.data(), rows, kOut, kIn, kOut);
      CHECK(axon::test::max_abs_diff(y, ysq) < 1e-5F);
    }
  }
}

TEST_CASE("a single-block quantized dot product is exact", "[kernels][quant]") {
  // Weight 1..32 scaled by d = 0.5, activation all ones quantized: the int dot is a known sum.
  axon::BlockQ8_0 w{};
  w.d = axon::fp32_to_fp16(0.5F);
  for (std::uint32_t i = 0; i < 32; ++i) {
    w.qs[i] = static_cast<std::int8_t>(i + 1);
  }
  std::vector<float> ones(32, 1.0F);
  float y = 0.0F;
  k().matmul_q8_0(&w, ones.data(), &y, 1, 32);
  CHECK(y == 0.5F * 528.0F);
  axon::BlockQ8_0 xq{};
  k().quantize_row_q8_0(ones.data(), &xq, 32);  // every element becomes 127 with d = 1/127
  k().matmul_q8_0_q8(&w, &xq, &y, 1, 32);
  const float dx = axon::fp16_to_fp32(xq.d);
  CHECK(y == static_cast<float>(528 * 127) * 0.5F * dx);
}

TEST_CASE("embedding rows dequantize Q8_0 and Q4_0 tables", "[kernels][quant]") {
  struct Case {
    const char* name;
    axon::DType type;
  };
  for (const auto& c : {Case{.name = "embedding_q8_0", .type = axon::DType::Q8_0},
                        Case{.name = "embedding_q4_0", .type = axon::DType::Q4_0},
                        Case{.name = "embedding_q4_1", .type = axon::DType::Q4_1},
                        Case{.name = "embedding_q6_k", .type = axon::DType::Q6_K}}) {
    const std::string base = c.name;
    const auto table = axon::test::read_fixture("quant/" + base + ".table.bin");
    const auto index = axon::test::load_axt(base + ".index", "quant").as_i32();
    const auto expected = axon::test::load_axt(base + ".rows", "quant").as_f32();
    const std::size_t n = c.type == axon::DType::Q6_K ? 256 : 64;
    std::vector<float> got(index.size() * n);
    for (std::size_t i = 0; i < index.size(); ++i) {
      k().embedding_row(c.type, table.data(), n, index[i], got.data() + i * n);
    }
    INFO(c.name);
    CHECK(got == expected);
  }
}
