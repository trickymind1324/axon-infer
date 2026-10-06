// Every NEON entry against the scalar reference. Integer paths and activation quantization
// must match bit for bit. Float paths differ by reduction order within a tolerance that was
// measured first (see the WARN lines) and then locked.

#if defined(AXON_HAVE_NEON)

#include "core/cpu_info.h"
#include "core/dtype.h"

#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

const axon::kernels::KernelTable& neon_table() {
  const auto table = axon::kernels::select(axon::kernels::KernelSet::Neon);
  if (!table) {
    SKIP(table.error().message);
  }
  return **table;
}

const axon::kernels::KernelTable& scalar() {
  return axon::kernels::scalar_kernels();
}

std::vector<float> random_floats(std::size_t n, std::uint32_t seed, float scale = 1.0F) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> out(n);
  for (auto& v : out) {
    v = dist(rng);
  }
  return out;
}

// |a - b| relative to the sum of |w_i x_i| for that output, the natural scale of rounding
// error in a dot product.
float relative_error(const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& w, const std::vector<float>& x, std::size_t n_out,
                     std::size_t n_in) {
  float worst = 0.0F;
  for (std::size_t o = 0; o < n_out; ++o) {
    float magnitude = 0.0F;
    for (std::size_t i = 0; i < n_in; ++i) {
      magnitude += std::abs(w[o * n_in + i] * x[i]);
    }
    worst = std::max(worst, std::abs(a[o] - b[o]) / (magnitude + 1e-30F));
  }
  return worst;
}

const std::size_t kInSizes[] = {32, 33, 64, 96, 100, 576, 1536, 2048};
const std::size_t kOutSizes[] = {1, 3, 7, 48, 576};

}  // namespace

TEST_CASE("NEON float matmuls stay within the locked tolerance of scalar", "[kernels][neon]") {
  const auto& neon = neon_table();
  float worst_f32 = 0.0F;
  float worst_f16 = 0.0F;
  for (const std::size_t n_in : kInSizes) {
    for (const std::size_t n_out : kOutSizes) {
      const auto w = random_floats(n_out * n_in, static_cast<std::uint32_t>(n_in * 7 + n_out));
      const auto x = random_floats(n_in, static_cast<std::uint32_t>(n_in + 1));
      std::vector<float> a(n_out);
      std::vector<float> b(n_out);
      scalar().matmul_f32(w.data(), x.data(), a.data(), n_out, n_in);
      neon.matmul_f32(w.data(), x.data(), b.data(), n_out, n_in);
      worst_f32 = std::max(worst_f32, relative_error(a, b, w, x, n_out, n_in));

      std::vector<std::uint16_t> half(w.size());
      std::vector<float> rounded(w.size());
      for (std::size_t i = 0; i < w.size(); ++i) {
        half[i] = axon::fp32_to_fp16(w[i]);
        rounded[i] = axon::fp16_to_fp32(half[i]);
      }
      scalar().matmul_f16(half.data(), x.data(), a.data(), n_out, n_in);
      neon.matmul_f16(half.data(), x.data(), b.data(), n_out, n_in);
      worst_f16 = std::max(worst_f16, relative_error(a, b, rounded, x, n_out, n_in));

      // The rows variants over three activation rows.
      const auto xs = random_floats(3 * n_in, static_cast<std::uint32_t>(n_in + 2));
      std::vector<float> ra(3 * n_out);
      std::vector<float> rb(3 * n_out);
      scalar().matmul_f32_rows(w.data(), xs.data(), ra.data(), 3, n_out, n_in);
      neon.matmul_f32_rows(w.data(), xs.data(), rb.data(), 3, n_out, n_in);
      for (std::size_t r = 0; r < 3; ++r) {
        const std::vector<float> pa(ra.begin() + static_cast<std::ptrdiff_t>(r * n_out),
                                    ra.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_out));
        const std::vector<float> pb(rb.begin() + static_cast<std::ptrdiff_t>(r * n_out),
                                    rb.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_out));
        const std::vector<float> xr(xs.begin() + static_cast<std::ptrdiff_t>(r * n_in),
                                    xs.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_in));
        worst_f32 = std::max(worst_f32, relative_error(pa, pb, w, xr, n_out, n_in));
      }
    }
  }
  // Measured on 2026-10-06 over these shapes: f32 1.97e-7, f16 1.53e-7. Locked at four times
  // the larger. A change that trips this is a bug or a different algorithm, not noise.
  WARN("NEON float matmul relative error against scalar: f32 " << worst_f32 << ", f16 "
                                                               << worst_f16);
  CHECK(worst_f32 < 8e-7F);
  CHECK(worst_f16 < 8e-7F);
}

TEST_CASE("NEON activation quantization and integer matmuls are bit-identical to scalar",
          "[kernels][neon]") {
  const auto& neon = neon_table();
  for (const std::size_t n_in : {std::size_t{32}, std::size_t{64}, std::size_t{96},
                                 std::size_t{576}, std::size_t{1536}, std::size_t{2048}}) {
    for (const std::size_t n_out : kOutSizes) {
      // Activations quantized through both tables, byte for byte.
      const auto x = random_floats(n_in, static_cast<std::uint32_t>(n_in + 3), 4.0F);
      std::vector<axon::BlockQ8_0> xs(n_in / 32);
      std::vector<axon::BlockQ8_0> xn(n_in / 32);
      scalar().quantize_row_q8_0(x.data(), xs.data(), n_in);
      neon.quantize_row_q8_0(x.data(), xn.data(), n_in);
      INFO("n_in " << n_in << " n_out " << n_out);
      CHECK(std::memcmp(xs.data(), xn.data(), xs.size() * sizeof(axon::BlockQ8_0)) == 0);

      // Weights quantized by the scalar quantizer for Q8_0, built by hand for the nibble formats.
      const auto w =
          random_floats(n_out * n_in, static_cast<std::uint32_t>(n_in * 11 + n_out), 0.5F);
      std::vector<axon::BlockQ8_0> w8(n_out * n_in / 32);
      scalar().quantize_row_q8_0(w.data(), w8.data(), w.size());
      std::vector<axon::BlockQ4_0> w4(w8.size());
      std::vector<axon::BlockQ4_1> w41(w8.size());
      std::mt19937 rng(static_cast<std::uint32_t>(n_in + n_out));
      for (std::size_t b = 0; b < w8.size(); ++b) {
        w4[b].d = axon::fp32_to_fp16(0.02F);
        w41[b].d = axon::fp32_to_fp16(0.02F);
        w41[b].m = axon::fp32_to_fp16(-0.17F);
        for (std::size_t i = 0; i < 16; ++i) {
          w4[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xFF);
          w41[b].qs[i] = static_cast<std::uint8_t>(rng() & 0xFF);
        }
      }
      std::vector<float> a(n_out);
      std::vector<float> b(n_out);
      scalar().matmul_q8_0_q8(w8.data(), xs.data(), a.data(), n_out, n_in);
      neon.matmul_q8_0_q8(w8.data(), xs.data(), b.data(), n_out, n_in);
      CHECK(a == b);
      scalar().matmul_q4_0_q8(w4.data(), xs.data(), a.data(), n_out, n_in);
      neon.matmul_q4_0_q8(w4.data(), xs.data(), b.data(), n_out, n_in);
      CHECK(a == b);
      scalar().matmul_q4_1_q8(w41.data(), xs.data(), a.data(), n_out, n_in);
      neon.matmul_q4_1_q8(w41.data(), xs.data(), b.data(), n_out, n_in);
      CHECK(a == b);

      // Rows variants over three activation rows.
      std::vector<axon::BlockQ8_0> xrows(3 * n_in / 32);
      for (std::size_t r = 0; r < 3; ++r) {
        const auto xr = random_floats(n_in, static_cast<std::uint32_t>(n_in + 10 + r), 4.0F);
        scalar().quantize_row_q8_0(xr.data(), xrows.data() + r * (n_in / 32), n_in);
      }
      std::vector<float> ra(3 * n_out);
      std::vector<float> rb(3 * n_out);
      scalar().matmul_q8_0_q8_rows(w8.data(), xrows.data(), ra.data(), 3, n_out, n_in);
      neon.matmul_q8_0_q8_rows(w8.data(), xrows.data(), rb.data(), 3, n_out, n_in);
      CHECK(ra == rb);
      scalar().matmul_q4_0_q8_rows(w4.data(), xrows.data(), ra.data(), 3, n_out, n_in);
      neon.matmul_q4_0_q8_rows(w4.data(), xrows.data(), rb.data(), 3, n_out, n_in);
      CHECK(ra == rb);
      scalar().matmul_q4_1_q8_rows(w41.data(), xrows.data(), ra.data(), 3, n_out, n_in);
      neon.matmul_q4_1_q8_rows(w41.data(), xrows.data(), rb.data(), 3, n_out, n_in);
      CHECK(ra == rb);
    }
  }
}

TEST_CASE("NEON Q6_K matmul stays within the locked tolerance of scalar", "[kernels][neon]") {
  const auto& neon = neon_table();
  float worst = 0.0F;
  for (const std::size_t n_in : {std::size_t{256}, std::size_t{512}, std::size_t{2048}}) {
    for (const std::size_t n_out :
         {std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{48}}) {
      // Random quants, high bits and scales with a small super-block scale, like the fixtures.
      std::vector<axon::BlockQ6_K> w(n_out * n_in / 256);
      std::mt19937 rng(static_cast<std::uint32_t>(n_in * 3 + n_out));
      for (auto& block : w) {
        for (auto& q : block.ql) {
          q = static_cast<std::uint8_t>(rng() & 0xFF);
        }
        for (auto& q : block.qh) {
          q = static_cast<std::uint8_t>(rng() & 0xFF);
        }
        for (auto& sc : block.scales) {
          sc = static_cast<std::int8_t>(rng() & 0xFF);
        }
        block.d = axon::fp32_to_fp16(1e-3F);
      }
      const auto x = random_floats(n_in, static_cast<std::uint32_t>(n_in + 5));
      std::vector<float> a(n_out);
      std::vector<float> b(n_out);
      scalar().matmul_q6_k(w.data(), x.data(), a.data(), n_out, n_in);
      neon.matmul_q6_k(w.data(), x.data(), b.data(), n_out, n_in);
      // Dequantized weights for the error scale.
      std::vector<float> deq(n_out * n_in);
      for (std::size_t i = 0; i < w.size(); ++i) {
        axon::dequantize_block_q6_k(w[i], deq.data() + i * 256);
      }
      worst = std::max(worst, relative_error(a, b, deq, x, n_out, n_in));
      const auto xs = random_floats(3 * n_in, static_cast<std::uint32_t>(n_in + 6));
      std::vector<float> ra(3 * n_out);
      std::vector<float> rb(3 * n_out);
      scalar().matmul_q6_k_rows(w.data(), xs.data(), ra.data(), 3, n_out, n_in);
      neon.matmul_q6_k_rows(w.data(), xs.data(), rb.data(), 3, n_out, n_in);
      for (std::size_t r = 0; r < 3; ++r) {
        const std::vector<float> pa(ra.begin() + static_cast<std::ptrdiff_t>(r * n_out),
                                    ra.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_out));
        const std::vector<float> pb(rb.begin() + static_cast<std::ptrdiff_t>(r * n_out),
                                    rb.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_out));
        const std::vector<float> xr(xs.begin() + static_cast<std::ptrdiff_t>(r * n_in),
                                    xs.begin() + static_cast<std::ptrdiff_t>((r + 1) * n_in));
        worst = std::max(worst, relative_error(pa, pb, deq, xr, n_out, n_in));
      }
    }
  }
  // Measured 2026-10-06: 1.54e-7. Locked at four times that, rounded up to match the float gates.
  WARN("NEON Q6_K matmul relative error against scalar: " << worst);
  CHECK(worst < 8e-7F);
}

#endif
