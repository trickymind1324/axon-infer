// The tiled multi-row kernels against the single-row kernel of the same table, row by row,
// bit for bit: a row's result must not depend on how many rows share the call. Row counts
// cover a tile, several tiles, and tails.

#include "core/dtype.h"

#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#if defined(AXON_HAVE_NEON)

namespace {

std::vector<float> random_floats(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

std::vector<axon::BlockQ6_K> random_q6_k(std::size_t n_blocks, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<axon::BlockQ6_K> w(n_blocks);
  for (auto& block : w) {
    for (auto& q : block.ql) {
      q = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& q : block.qh) {
      q = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& sc : block.scales) {
      sc = static_cast<std::int8_t>(static_cast<int>(rng() % 64) - 32);
    }
    block.d = axon::fp32_to_fp16(0.01F);
  }
  return w;
}

template <class Rows, class Single>
void check_rows(const std::string& name, std::size_t n_rows, std::size_t n_out, Rows rows,
                Single single) {
  std::vector<float> tiled(n_rows * n_out);
  rows(tiled.data());
  std::vector<float> expected(n_rows * n_out);
  for (std::size_t r = 0; r < n_rows; ++r) {
    single(r, expected.data() + r * n_out);
  }
  INFO(name << " rows " << n_rows << " out " << n_out);
  CHECK(tiled == expected);
}

}  // namespace

TEST_CASE("tiled multi-row kernels equal the single-row kernel row by row",
          "[kernels][neon][rows]") {
  const auto selected = axon::kernels::select(axon::kernels::KernelSet::Neon);
  if (!selected) {
    SKIP("no NEON table on this host");
  }
  const auto& k = **selected;
  for (const std::size_t n_rows :
       {std::size_t{1}, std::size_t{4}, std::size_t{5}, std::size_t{9}}) {
    for (const std::size_t n_out : {std::size_t{1}, std::size_t{7}}) {
      const std::size_t n_in = 512;
      const auto x =
          random_floats(n_rows * n_in, static_cast<std::uint32_t>(n_rows * 7 + n_out), 2.0F);
      const auto w6 = random_q6_k(n_out * n_in / axon::kQK_K, static_cast<std::uint32_t>(n_out));
      check_rows(
          "q6_k", n_rows, n_out,
          [&](float* y) { k.matmul_q6_k_rows(w6.data(), x.data(), y, n_rows, n_out, n_in, n_out); },
          [&](std::size_t r, float* y) {
            k.matmul_q6_k(w6.data(), x.data() + r * n_in, y, n_out, n_in);
          });
    }
  }
}

#endif
