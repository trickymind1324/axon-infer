// The multi-row matmuls with an output stride: every table's `_rows` entry, on a column chunk
// narrower than the stride, writes exactly the single-row kernel's values into its own columns
// and nothing into the columns between them. The row and column counts are chosen so the
// tiled kernels see tails.

#include "core/dtype.h"

#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kRows = 5;
constexpr std::size_t kOut = 7;
constexpr std::size_t kStride = 11;
constexpr float kSentinel = -12345.0F;

std::vector<float> random_floats(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

std::vector<std::uint16_t> halves(const std::vector<float>& v) {
  std::vector<std::uint16_t> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    h[i] = axon::fp32_to_fp16(v[i]);
  }
  return h;
}

std::vector<std::uint16_t> bfloats(const std::vector<float>& v) {
  std::vector<std::uint16_t> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    h[i] = axon::fp32_to_bf16(v[i]);
  }
  return h;
}

// Runs `rows` into a strided buffer and `single` per row into a contiguous one, and checks the
// strided buffer holds the single-row values in its columns and the sentinel elsewhere.
template <class Rows, class Single>
void check_strided(const std::string& name, std::size_t n_in, Rows rows, Single single) {
  std::vector<float> strided(kRows * kStride, kSentinel);
  rows(strided.data(), kStride);
  std::vector<float> expected(kRows * kOut);
  for (std::size_t r = 0; r < kRows; ++r) {
    single(r, expected.data() + r * kOut);
  }
  INFO(name << " n_in " << n_in);
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t c = 0; c < kStride; ++c) {
      const float got = strided[r * kStride + c];
      if (c < kOut) {
        CHECK(got == expected[r * kOut + c]);
      } else {
        CHECK(got == kSentinel);
      }
    }
  }
}

void check_table(const axon::kernels::KernelTable& k, const std::string& table) {
  for (const std::size_t n_in : {std::size_t{64}, std::size_t{256}}) {
    const auto w = random_floats(kOut * n_in, static_cast<std::uint32_t>(n_in), 0.5F);
    const auto x = random_floats(kRows * n_in, static_cast<std::uint32_t>(n_in + 1), 2.0F);
    const auto w16 = halves(w);
    const auto wbf = bfloats(w);
    std::vector<axon::BlockQ8_0> w8(kOut * n_in / 32);
    k.quantize_row_q8_0(w.data(), w8.data(), w.size());
    std::vector<axon::BlockQ8_0> xq(kRows * n_in / 32);
    for (std::size_t r = 0; r < kRows; ++r) {
      k.quantize_row_q8_0(x.data() + r * n_in, xq.data() + r * (n_in / 32), n_in);
    }
    std::mt19937 rng(static_cast<std::uint32_t>(n_in + 7));
    std::vector<axon::BlockQ4_0> w4(w8.size());
    std::vector<axon::BlockQ4_1> w41(w8.size());
    for (std::size_t b = 0; b < w8.size(); ++b) {
      w4[b].d = axon::fp32_to_fp16(0.02F);
      w41[b].d = axon::fp32_to_fp16(0.02F);
      w41[b].m = axon::fp32_to_fp16(-0.17F);
      for (std::size_t i = 0; i < 16; ++i) {
        w4[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xFF);
        w41[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xFF);
      }
    }
    const std::size_t nb = n_in / 32;
    const std::string t = table + " ";

    check_strided(
        t + "f32", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_f32_rows(w.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_f32(w.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "f16", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_f16_rows(w16.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_f16(w16.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "bf16", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_bf16_rows(wbf.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_bf16(wbf.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "q8_0 float", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q8_0_rows(w8.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q8_0(w8.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "q8_0 q8", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q8_0_q8_rows(w8.data(), xq.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q8_0_q8(w8.data(), xq.data() + r * nb, y, kOut, n_in);
        });
    check_strided(
        t + "q4_0 float", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q4_0_rows(w4.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q4_0(w4.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "q4_0 q8", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q4_0_q8_rows(w4.data(), xq.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q4_0_q8(w4.data(), xq.data() + r * nb, y, kOut, n_in);
        });
    check_strided(
        t + "q4_1 float", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q4_1_rows(w41.data(), x.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q4_1(w41.data(), x.data() + r * n_in, y, kOut, n_in);
        });
    check_strided(
        t + "q4_1 q8", n_in,
        [&](float* y, std::size_t s) {
          k.matmul_q4_1_q8_rows(w41.data(), xq.data(), y, kRows, kOut, n_in, s);
        },
        [&](std::size_t r, float* y) {
          k.matmul_q4_1_q8(w41.data(), xq.data() + r * nb, y, kOut, n_in);
        });
    if (n_in % axon::kQK_K == 0) {
      std::vector<axon::BlockQ6_K> w6(kOut * n_in / axon::kQK_K);
      for (auto& block : w6) {
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
      check_strided(
          t + "q6_k", n_in,
          [&](float* y, std::size_t s) {
            k.matmul_q6_k_rows(w6.data(), x.data(), y, kRows, kOut, n_in, s);
          },
          [&](std::size_t r, float* y) {
            k.matmul_q6_k(w6.data(), x.data() + r * n_in, y, kOut, n_in);
          });
    }
  }
}

}  // namespace

TEST_CASE("multi-row matmuls honour the output stride on every table", "[kernels][rows]") {
  check_table(axon::kernels::scalar_kernels(), "scalar");
#if defined(AXON_HAVE_NEON)
  const auto neon = axon::kernels::select(axon::kernels::KernelSet::Neon);
  if (neon) {
    check_table(**neon, "neon");
  }
#endif
}
