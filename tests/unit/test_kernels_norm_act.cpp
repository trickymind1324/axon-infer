#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

const axon::kernels::KernelTable& k() {
  return axon::kernels::kernels();
}

}  // namespace

TEST_CASE("axt fixture reader", "[kernels][fixture]") {
  const auto x = axon::test::load_axt("rmsnorm_64.x");
  REQUIRE(x.dims.size() == 1);
  CHECK(x.dim(0) == 64);
  CHECK(x.type == axon::test::AxtType::F32);
  CHECK(x.numel() == 64);
  CHECK(x.as_f32().size() == 64);

  const auto half = axon::test::load_axt("matmul_f16.w");
  REQUIRE(half.dims.size() == 2);
  CHECK(half.dim(0) == 48);
  CHECK(half.dim(1) == 64);
  CHECK(half.type == axon::test::AxtType::F16);
  CHECK(half.as_u16().size() == std::size_t{48} * 64);
  const auto values = half.as_f32();
  for (const float v : values) {
    CHECK(std::abs(v) <= 0.125F);
  }

  const auto index = axon::test::load_axt("embedding_f32.index");
  CHECK(index.type == axon::test::AxtType::I32);
  CHECK(index.as_i32() == std::vector<std::int32_t>{0, 7, 15});
}

TEST_CASE("scalar table is the active table", "[kernels]") {
  CHECK(&axon::kernels::kernels() == &axon::kernels::scalar_kernels());
  CHECK(axon::kernels::kernels().matmul_f32 != nullptr);
  CHECK(axon::kernels::kernels().attention != nullptr);
}

TEST_CASE("rmsnorm matches the fixture", "[kernels][rmsnorm]") {
  for (const char* n : {"64", "1"}) {
    const std::string base = std::string("rmsnorm_") + n;
    const auto x = axon::test::load_axt(base + ".x").as_f32();
    const auto w = axon::test::load_axt(base + ".w").as_f32();
    const auto expected = axon::test::load_axt(base + ".y").as_f32();
    std::vector<float> y(x.size());
    k().rmsnorm(x.data(), w.data(), y.data(), x.size(), 1e-5F);
    INFO("n = " << n);
    CHECK(axon::test::max_abs_diff(y, expected) < 1e-5F);
  }
}

TEST_CASE("rmsnorm of a constant vector returns the weight", "[kernels][rmsnorm]") {
  const std::size_t n = 37;
  std::vector<float> x(n, 4.5F);
  std::vector<float> w(n);
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = static_cast<float>(i) - 10.0F;
  }
  std::vector<float> y(n);
  k().rmsnorm(x.data(), w.data(), y.data(), n, 0.0F);
  CHECK(axon::test::max_abs_diff(y, w) < 1e-5F);
}

TEST_CASE("softmax matches the fixture", "[kernels][softmax]") {
  for (const char* n : {"1", "5", "64"}) {
    const std::string base = std::string("softmax_") + n;
    auto x = axon::test::load_axt(base + ".x").as_f32();
    const auto expected = axon::test::load_axt(base + ".y").as_f32();
    k().softmax(x.data(), x.size());
    INFO("n = " << n);
    CHECK(axon::test::max_abs_diff(x, expected) < 1e-5F);
  }
}

TEST_CASE("softmax sums to one and ignores a constant shift", "[kernels][softmax]") {
  std::vector<float> x{3.0F, -1.0F, 0.5F, 8.0F, 8.0F, -20.0F, 2.25F};
  std::vector<float> shifted(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    shifted[i] = x[i] + 1000.0F;
  }
  k().softmax(x.data(), x.size());
  k().softmax(shifted.data(), shifted.size());
  double sum = 0.0;
  for (const float v : x) {
    CHECK(v >= 0.0F);
    sum += static_cast<double>(v);
  }
  CHECK(std::abs(sum - 1.0) < 1e-6);
  CHECK(x[3] == x[4]);
  CHECK(axon::test::max_abs_diff(x, shifted) < 1e-6F);

  std::vector<float> one{123.0F};
  k().softmax(one.data(), 1);
  CHECK(one[0] == 1.0F);
}

TEST_CASE("silu_mul matches the fixture", "[kernels][silu]") {
  const auto gate = axon::test::load_axt("silu_mul.gate").as_f32();
  const auto up = axon::test::load_axt("silu_mul.up").as_f32();
  const auto expected = axon::test::load_axt("silu_mul.out").as_f32();
  std::vector<float> out(gate.size());
  k().silu_mul(gate.data(), up.data(), out.data(), gate.size());
  CHECK(axon::test::max_abs_diff(out, expected) < 1e-5F);
}

TEST_CASE("silu is zero at zero and linear for large inputs", "[kernels][silu]") {
  const std::vector<float> gate{0.0F, 40.0F, -40.0F};
  const std::vector<float> up{7.0F, 2.0F, 2.0F};
  std::vector<float> out(3);
  k().silu_mul(gate.data(), up.data(), out.data(), 3);
  CHECK(out[0] == 0.0F);
  CHECK(std::abs(out[1] - 80.0F) < 1e-4F);
  CHECK(std::abs(out[2]) < 1e-6F);
}
