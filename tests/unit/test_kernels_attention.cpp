#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

const axon::kernels::KernelTable& k() {
  return axon::kernels::kernels();
}

struct Shape {
  const char* name;
  std::size_t seq;
  std::size_t n_head;
  std::size_t n_head_kv;
  std::size_t head_dim;
};

constexpr Shape kShapes[] = {
    {.name = "attention_s8_h4_kv2_d16", .seq = 8, .n_head = 4, .n_head_kv = 2, .head_dim = 16},
    {.name = "attention_s1_h2_kv1_d8", .seq = 1, .n_head = 2, .n_head_kv = 1, .head_dim = 8},
    {.name = "attention_s5_h2_kv1_d8", .seq = 5, .n_head = 2, .n_head_kv = 1, .head_dim = 8},
};

float inv_sqrt(std::size_t n) {
  return 1.0F / std::sqrt(static_cast<float>(n));
}

}  // namespace

TEST_CASE("attention matches the fixtures", "[kernels][attention]") {
  for (const auto& s : kShapes) {
    const std::string base = s.name;
    const auto q = axon::test::load_axt(base + ".q").as_f32();
    const auto key = axon::test::load_axt(base + ".k").as_f32();
    const auto v = axon::test::load_axt(base + ".v").as_f32();
    const auto expected = axon::test::load_axt(base + ".out").as_f32();
    REQUIRE(q.size() == s.seq * s.n_head * s.head_dim);
    REQUIRE(key.size() == s.seq * s.n_head_kv * s.head_dim);
    std::vector<float> out(q.size());
    std::vector<float> scores(s.seq * s.seq);
    k().attention(q.data(), key.data(), v.data(), out.data(), scores.data(), s.seq, s.n_head,
                  s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    INFO(s.name);
    CHECK(axon::test::max_abs_diff(out, expected) < 1e-5F);
  }
}

TEST_CASE("attention over one position returns the value of the mapped kv head",
          "[kernels][attention]") {
  const std::size_t n_head = 4;
  const std::size_t n_head_kv = 2;
  const std::size_t head_dim = 8;
  std::vector<float> q(n_head * head_dim, 0.3F);
  std::vector<float> key(n_head_kv * head_dim, -0.7F);
  std::vector<float> v(n_head_kv * head_dim);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>(i);
  }
  std::vector<float> out(q.size());
  float scores = 0.0F;
  k().attention(q.data(), key.data(), v.data(), out.data(), &scores, 1, n_head, n_head_kv, head_dim,
                inv_sqrt(head_dim));
  for (std::size_t h = 0; h < n_head; ++h) {
    const std::size_t kv_head = h / (n_head / n_head_kv);
    for (std::size_t d = 0; d < head_dim; ++d) {
      CHECK(out[h * head_dim + d] == v[kv_head * head_dim + d]);
    }
  }
}

TEST_CASE("attention with zero queries averages the visible values", "[kernels][attention]") {
  const std::size_t seq = 6;
  const std::size_t head_dim = 4;
  std::vector<float> q(seq * head_dim, 0.0F);
  std::vector<float> key(seq * head_dim, 1.0F);
  std::vector<float> v(seq * head_dim);
  for (std::size_t i = 0; i < seq; ++i) {
    for (std::size_t d = 0; d < head_dim; ++d) {
      v[i * head_dim + d] = static_cast<float>(i) * 10.0F + static_cast<float>(d);
    }
  }
  std::vector<float> out(q.size());
  std::vector<float> scores(seq * seq);
  k().attention(q.data(), key.data(), v.data(), out.data(), scores.data(), seq, 1, 1, head_dim,
                inv_sqrt(head_dim));
  for (std::size_t i = 0; i < seq; ++i) {
    for (std::size_t d = 0; d < head_dim; ++d) {
      // Mean of v[0..i] along d: 10 * i / 2 + d.
      const float expected = 5.0F * static_cast<float>(i) + static_cast<float>(d);
      CHECK(std::abs(out[i * head_dim + d] - expected) < 1e-4F);
    }
  }
}

TEST_CASE("grouped-query attention equals multi-head attention on repeated kv heads",
          "[kernels][attention]") {
  const Shape& s = kShapes[0];
  const std::string base = s.name;
  const auto q = axon::test::load_axt(base + ".q").as_f32();
  const auto key = axon::test::load_axt(base + ".k").as_f32();
  const auto v = axon::test::load_axt(base + ".v").as_f32();
  const std::size_t group = s.n_head / s.n_head_kv;

  // Expand k and v so every query head has its own copy of the kv head it maps to.
  std::vector<float> key_full(s.seq * s.n_head * s.head_dim);
  std::vector<float> v_full(key_full.size());
  for (std::size_t i = 0; i < s.seq; ++i) {
    for (std::size_t h = 0; h < s.n_head; ++h) {
      const std::size_t src = (i * s.n_head_kv + h / group) * s.head_dim;
      const std::size_t dst = (i * s.n_head + h) * s.head_dim;
      for (std::size_t d = 0; d < s.head_dim; ++d) {
        key_full[dst + d] = key[src + d];
        v_full[dst + d] = v[src + d];
      }
    }
  }

  std::vector<float> out_gqa(q.size());
  std::vector<float> out_mha(q.size());
  std::vector<float> scores(s.seq * s.seq);
  k().attention(q.data(), key.data(), v.data(), out_gqa.data(), scores.data(), s.seq, s.n_head,
                s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
  k().attention(q.data(), key_full.data(), v_full.data(), out_mha.data(), scores.data(), s.seq,
                s.n_head, s.n_head, s.head_dim, inv_sqrt(s.head_dim));
  CHECK(out_gqa == out_mha);
}

TEST_CASE("attention_kv with nothing cached equals full attention", "[kernels][attention]") {
  for (const auto& s : kShapes) {
    const std::string base = s.name;
    const auto q = axon::test::load_axt(base + ".q").as_f32();
    const auto key = axon::test::load_axt(base + ".k").as_f32();
    const auto v = axon::test::load_axt(base + ".v").as_f32();
    std::vector<float> full(q.size());
    std::vector<float> cached(q.size());
    std::vector<float> scores(s.seq * s.seq);
    k().attention(q.data(), key.data(), v.data(), full.data(), scores.data(), s.seq, s.n_head,
                  s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    k().attention_kv(q.data(), key.data(), v.data(), cached.data(), scores.data(), s.seq, 0,
                     s.n_head, s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    INFO(s.name);
    CHECK(cached == full);
  }
}

TEST_CASE("attention_kv over a growing cache is bit-identical to the full pass",
          "[kernels][attention]") {
  const Shape& s = kShapes[0];  // seq 8
  const std::string base = s.name;
  const auto q = axon::test::load_axt(base + ".q").as_f32();
  const auto key = axon::test::load_axt(base + ".k").as_f32();
  const auto v = axon::test::load_axt(base + ".v").as_f32();
  const std::size_t q_row = s.n_head * s.head_dim;
  std::vector<float> full(q.size());
  std::vector<float> scores(s.seq * s.seq);
  k().attention(q.data(), key.data(), v.data(), full.data(), scores.data(), s.seq, s.n_head,
                s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));

  SECTION("prefill five positions, then one at a time") {
    std::vector<float> out(q.size());
    k().attention_kv(q.data(), key.data(), v.data(), out.data(), scores.data(), 5, 0, s.n_head,
                     s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    for (std::size_t pos = 5; pos < s.seq; ++pos) {
      k().attention_kv(q.data() + pos * q_row, key.data(), v.data(), out.data() + pos * q_row,
                       scores.data(), 1, pos, s.n_head, s.n_head_kv, s.head_dim,
                       inv_sqrt(s.head_dim));
    }
    CHECK(out == full);
  }
  SECTION("two prefill chunks of three and five") {
    std::vector<float> out(q.size());
    k().attention_kv(q.data(), key.data(), v.data(), out.data(), scores.data(), 3, 0, s.n_head,
                     s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    k().attention_kv(q.data() + 3 * q_row, key.data(), v.data(), out.data() + 3 * q_row,
                     scores.data(), 5, 3, s.n_head, s.n_head_kv, s.head_dim, inv_sqrt(s.head_dim));
    CHECK(out == full);
  }
}
