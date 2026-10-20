// The NEON F32 attention against the scalar kernel: the scores differ by the dot product's
// reduction order within a tolerance measured first and then locked, the output accumulation
// is the same fused chain per element. A row's result does not depend on which rows or heads
// share the call, which is what the model's head by row-group split relies on.

#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
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

struct Shape {
  std::size_t n_q;
  std::size_t n_past;
  std::size_t n_head;
  std::size_t n_head_kv;
  std::size_t head_dim;
};

struct Run {
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> out;
  std::vector<float> scores;
};

Run make(const Shape& s, std::uint32_t seed) {
  const std::size_t total = s.n_past + s.n_q;
  Run r;
  r.q = random_floats(s.n_q * s.n_head * s.head_dim, seed, 1.0F);
  r.k = random_floats(total * s.n_head_kv * s.head_dim, seed + 1, 1.0F);
  r.v = random_floats(total * s.n_head_kv * s.head_dim, seed + 2, 1.0F);
  r.out.assign(s.n_q * s.n_head * s.head_dim, 0.0F);
  r.scores.assign(s.n_q * total, 0.0F);
  return r;
}

void run(const axon::kernels::KernelTable& k, const Shape& s, Run& r, std::size_t head_begin,
         std::size_t head_end) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(s.head_dim));
  k.attention_kv_heads(r.q.data(), r.k.data(), r.v.data(), r.out.data(), r.scores.data(), s.n_q,
                       s.n_past, s.n_head, s.n_head_kv, s.head_dim, scale, head_begin, head_end);
}

float ratio(const std::vector<float>& a, const std::vector<float>& b) {
  float diff = 0.0F;
  float range = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(a[i] - b[i]));
    range = std::max(range, std::abs(a[i]));
  }
  return diff / range;
}

}  // namespace

TEST_CASE("NEON F32 attention stays within the locked tolerance of scalar",
          "[kernels][neon][attention]") {
  const auto selected = axon::kernels::select(axon::kernels::KernelSet::Neon);
  if (!selected) {
    SKIP("no NEON table on this host");
  }
  const auto& neon = **selected;
  const auto& scalar = axon::kernels::scalar_kernels();
  float worst = 0.0F;
  // SmolLM2's heads, a wide head, and head sizes that leave four-wide and scalar tails.
  for (const Shape& s :
       {Shape{.n_q = 5, .n_past = 37, .n_head = 9, .n_head_kv = 3, .head_dim = 64},
        Shape{.n_q = 1, .n_past = 200, .n_head = 4, .n_head_kv = 4, .head_dim = 128},
        Shape{.n_q = 3, .n_past = 6, .n_head = 2, .n_head_kv = 1, .head_dim = 40},
        Shape{.n_q = 2, .n_past = 0, .n_head = 2, .n_head_kv = 2, .head_dim = 37}}) {
    Run a = make(s, static_cast<std::uint32_t>(s.head_dim));
    Run b = a;
    run(scalar, s, a, 0, s.n_head);
    run(neon, s, b, 0, s.n_head);
    INFO("head_dim " << s.head_dim);
    worst = std::max(worst, ratio(a.out, b.out));
  }
  // Measured 2026-10-20: 2.67e-7 over these four shapes. Locked at four times the measurement.
  INFO("NEON F32 attention against scalar, max abs output difference over the output range: "
       << worst);
  CHECK(worst < 1.1e-6F);
}

TEST_CASE("NEON F32 attention rows do not depend on the rows and heads that share the call",
          "[kernels][neon][attention]") {
  const auto selected = axon::kernels::select(axon::kernels::KernelSet::Neon);
  if (!selected) {
    SKIP("no NEON table on this host");
  }
  const auto& neon = **selected;
  const Shape s{.n_q = 7, .n_past = 20, .n_head = 6, .n_head_kv = 2, .head_dim = 64};
  Run whole = make(s, 99);
  run(neon, s, whole, 0, s.n_head);

  // Rows [3, 7) of heads [2, 5) through offset pointers, as the model's split does.
  Run part = make(s, 99);
  const std::size_t first = 3;
  const std::size_t rows = s.n_q - first;
  const std::size_t q_row = s.n_head * s.head_dim;
  const float scale = 1.0F / std::sqrt(static_cast<float>(s.head_dim));
  neon.attention_kv_heads(part.q.data() + first * q_row, part.k.data(), part.v.data(),
                          part.out.data() + first * q_row, part.scores.data(), rows,
                          s.n_past + first, s.n_head, s.n_head_kv, s.head_dim, scale, 2, 5);
  for (std::size_t i = first; i < s.n_q; ++i) {
    for (std::size_t h = 2; h < 5; ++h) {
      const std::size_t at = (i * s.n_head + h) * s.head_dim;
      for (std::size_t d = 0; d < s.head_dim; ++d) {
        REQUIRE(part.out[at + d] == whole.out[at + d]);
      }
    }
  }
}

#endif
