#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

using axon::kernels::RopeParams;
using axon::kernels::RopeStyle;

const axon::kernels::KernelTable& k() {
  return axon::kernels::kernels();
}

void check_fixture(const std::string& base, RopeStyle style, float theta, bool with_factors) {
  const auto x = axon::test::load_axt(base + ".x");
  const auto values = x.as_f32();
  const auto positions = axon::test::load_axt(base + ".pos").as_i32();
  const auto expected = axon::test::load_axt(base + ".y").as_f32();
  std::vector<float> factors;
  if (with_factors) {
    factors = axon::test::load_axt(base + ".factors").as_f32();
    REQUIRE(factors.size() == x.dim(2) / 2);
  }
  const std::size_t n_heads = x.dim(1);
  const std::size_t head_dim = x.dim(2);
  REQUIRE(x.dim(0) == positions.size());
  const RopeParams params{.head_dim = head_dim,
                          .rotary_dim = head_dim,
                          .theta = theta,
                          .style = style,
                          .freq_factors = with_factors ? factors.data() : nullptr};
  std::vector<float> got = values;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    k().rope(got.data() + i * n_heads * head_dim, n_heads, static_cast<std::size_t>(positions[i]),
             params);
  }
  INFO(base);
  CHECK(axon::test::max_abs_diff(got, expected) < 1e-5F);
}

}  // namespace

TEST_CASE("rope matches the fixtures", "[kernels][rope]") {
  check_fixture("rope_normal", RopeStyle::Normal, 10000.0F, false);
  check_fixture("rope_neox", RopeStyle::NeoX, 1000000.0F, false);
  check_fixture("rope_llama3", RopeStyle::Normal, 500000.0F, true);
}

TEST_CASE("rope at position zero is the identity", "[kernels][rope]") {
  const auto x = axon::test::load_axt("rope_normal.x");
  const auto values = x.as_f32();
  for (const RopeStyle style : {RopeStyle::Normal, RopeStyle::NeoX}) {
    std::vector<float> got = values;
    const RopeParams params{.head_dim = x.dim(2),
                            .rotary_dim = x.dim(2),
                            .theta = 10000.0F,
                            .style = style,
                            .freq_factors = nullptr};
    k().rope(got.data(), x.dim(0) * x.dim(1), 0, params);
    CHECK(got == values);
  }
}

TEST_CASE("rope preserves the norm of every rotated pair", "[kernels][rope]") {
  const std::size_t head_dim = 16;
  const std::size_t n_heads = 3;
  std::vector<float> x(n_heads * head_dim);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(i % 7) - 3.0F;
  }
  for (const RopeStyle style : {RopeStyle::Normal, RopeStyle::NeoX}) {
    std::vector<float> got = x;
    const RopeParams params{.head_dim = head_dim,
                            .rotary_dim = head_dim,
                            .theta = 10000.0F,
                            .style = style,
                            .freq_factors = nullptr};
    k().rope(got.data(), n_heads, 1234, params);
    for (std::size_t h = 0; h < n_heads; ++h) {
      for (std::size_t i = 0; i < head_dim / 2; ++i) {
        const std::size_t a = style == RopeStyle::Normal ? 2 * i : i;
        const std::size_t b = style == RopeStyle::Normal ? 2 * i + 1 : i + head_dim / 2;
        const float* in = x.data() + h * head_dim;
        const float* out = got.data() + h * head_dim;
        const float before = in[a] * in[a] + in[b] * in[b];
        const float after = out[a] * out[a] + out[b] * out[b];
        CHECK(std::abs(before - after) < 1e-4F);
      }
    }
  }
}

TEST_CASE("rope leaves dimensions past rotary_dim untouched", "[kernels][rope]") {
  std::vector<float> x(8, 1.0F);
  const RopeParams params{.head_dim = 8,
                          .rotary_dim = 4,
                          .theta = 10000.0F,
                          .style = RopeStyle::NeoX,
                          .freq_factors = nullptr};
  k().rope(x.data(), 1, 5, params);
  for (std::size_t i = 4; i < 8; ++i) {
    CHECK(x[i] == 1.0F);
  }
  CHECK(x[0] != 1.0F);
}

TEST_CASE("rope with unit frequency factors equals plain rope", "[kernels][rope]") {
  const auto x = axon::test::load_axt("rope_neox.x").as_f32();
  const std::vector<float> ones(8, 1.0F);
  std::vector<float> plain = x;
  std::vector<float> scaled = x;
  RopeParams params{.head_dim = 16,
                    .rotary_dim = 16,
                    .theta = 500000.0F,
                    .style = RopeStyle::Normal,
                    .freq_factors = nullptr};
  k().rope(plain.data(), 4, 77, params);
  params.freq_factors = ones.data();
  k().rope(scaled.data(), 4, 77, params);
  CHECK(plain == scaled);
}
