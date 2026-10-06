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

#endif
