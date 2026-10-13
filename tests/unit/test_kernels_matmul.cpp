#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace {

const axon::kernels::KernelTable& k() {
  return axon::kernels::kernels();
}

}  // namespace

TEST_CASE("matmul_f32 matches the fixture", "[kernels][matmul]") {
  for (const char* name : {"matmul_f32", "matmul_f32_odd"}) {
    const std::string base = name;
    const auto w = axon::test::load_axt(base + ".w");
    const auto weights = w.as_f32();
    const auto x = axon::test::load_axt(base + ".x").as_f32();
    const auto expected = axon::test::load_axt(base + ".y").as_f32();
    const std::size_t n_out = w.dim(0);
    const std::size_t n_in = w.dim(1);
    REQUIRE(x.size() == n_in);
    std::vector<float> y(n_out);
    k().matmul_f32(weights.data(), x.data(), y.data(), n_out, n_in);
    INFO(name);
    CHECK(axon::test::max_abs_diff(y, expected) < 1e-5F);
  }
}

TEST_CASE("matmul_f16 and matmul_bf16 match the fixtures", "[kernels][matmul]") {
  for (const char* name : {"matmul_f16", "matmul_bf16"}) {
    const std::string base = name;
    const auto w = axon::test::load_axt(base + ".w");
    const auto weights = w.as_u16();
    const auto x = axon::test::load_axt(base + ".x").as_f32();
    const auto expected = axon::test::load_axt(base + ".y").as_f32();
    std::vector<float> y(w.dim(0));
    if (base == "matmul_f16") {
      k().matmul_f16(weights.data(), x.data(), y.data(), w.dim(0), w.dim(1));
    } else {
      k().matmul_bf16(weights.data(), x.data(), y.data(), w.dim(0), w.dim(1));
    }
    INFO(name);
    // The reference uses the rounded weights too, so only float accumulation error remains.
    CHECK(axon::test::max_abs_diff(y, expected) < 1e-5F);
  }
}

TEST_CASE("row matmuls match the fixtures", "[kernels][matmul]") {
  for (const char* name : {"matmul_f32", "matmul_f16", "matmul_bf16"}) {
    const std::string base = name;
    const auto w = axon::test::load_axt(base + ".w");
    const auto x = axon::test::load_axt(base + "_rows.x");
    const auto rows = x.as_f32();
    const auto expected = axon::test::load_axt(base + "_rows.y").as_f32();
    const std::size_t n_rows = x.dim(0);
    const std::size_t n_out = w.dim(0);
    const std::size_t n_in = w.dim(1);
    REQUIRE(x.dim(1) == n_in);
    std::vector<float> y(n_rows * n_out);
    if (base == "matmul_f32") {
      const auto weights = w.as_f32();
      k().matmul_f32_rows(weights.data(), rows.data(), y.data(), n_rows, n_out, n_in, n_out);
    } else if (base == "matmul_f16") {
      const auto weights = w.as_u16();
      k().matmul_f16_rows(weights.data(), rows.data(), y.data(), n_rows, n_out, n_in, n_out);
    } else {
      const auto weights = w.as_u16();
      k().matmul_bf16_rows(weights.data(), rows.data(), y.data(), n_rows, n_out, n_in, n_out);
    }
    INFO(name);
    CHECK(axon::test::max_abs_diff(y, expected) < 1e-5F);
  }
}

TEST_CASE("matmul with a single output row is a dot product", "[kernels][matmul]") {
  const std::vector<float> w{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  const std::vector<float> x{0.5F, -1.0F, 2.0F, 0.0F, 1.0F};
  float y = 0.0F;
  k().matmul_f32(w.data(), x.data(), &y, 1, 5);
  CHECK(y == 0.5F - 2.0F + 6.0F + 5.0F);
}

TEST_CASE("matmul with no inputs writes zeros", "[kernels][matmul]") {
  std::vector<float> y{9.0F, 9.0F};
  k().matmul_f32(nullptr, nullptr, y.data(), 2, 0);
  CHECK(y == std::vector<float>{0.0F, 0.0F});
}
