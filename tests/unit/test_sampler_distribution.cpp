// The sampler's distribution and draw against sample: the distribution is what sample draws
// from, normalised with the right support under every stage, and a draw from it with the same
// generator state picks the token sample picks.

#include "axon/sampler.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr std::int32_t kVocab = 97;

std::vector<float> random_logits(std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0F, 3.0F);
  std::vector<float> v(static_cast<std::size_t>(kVocab));
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

}  // namespace

TEST_CASE("the distribution is normalised with the support each stage leaves",
          "[sampler][distribution]") {
  const std::vector<std::int32_t> recent{3, 5, 5, 8};
  struct Case {
    axon::SamplerConfig config;
    std::size_t max_support;  // what the stages can keep at most
  };
  for (const auto& c : {Case{.config = {.temperature = 0.0F, .seed = 1}, .max_support = 1},
                        Case{.config = {.temperature = 1.0F,
                                        .top_k = 0,
                                        .top_p = 1.0F,
                                        .min_p = 0.0F,
                                        .repeat_penalty = 1.0F,
                                        .seed = 1},
                             .max_support = static_cast<std::size_t>(kVocab)},
                        Case{.config = {.temperature = 0.7F,
                                        .top_k = 5,
                                        .top_p = 1.0F,
                                        .min_p = 0.0F,
                                        .repeat_penalty = 1.0F,
                                        .seed = 1},
                             .max_support = 5},
                        Case{.config = {.temperature = 1.0F,
                                        .top_k = 0,
                                        .top_p = 0.5F,
                                        .min_p = 0.0F,
                                        .repeat_penalty = 1.0F,
                                        .seed = 1},
                             .max_support = static_cast<std::size_t>(kVocab)},
                        Case{.config = {.temperature = 1.0F,
                                        .top_k = 0,
                                        .top_p = 1.0F,
                                        .min_p = 0.3F,
                                        .repeat_penalty = 1.3F,
                                        .seed = 1},
                             .max_support = static_cast<std::size_t>(kVocab)}}) {
    axon::Sampler sampler(c.config, kVocab);
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
      const auto logits = random_logits(seed);
      std::vector<float> p(static_cast<std::size_t>(kVocab));
      sampler.distribution(logits, recent, p);
      double sum = 0.0;
      std::size_t support = 0;
      for (const float x : p) {
        REQUIRE(x >= 0.0F);
        sum += static_cast<double>(x);
        support += x > 0.0F ? 1 : 0;
      }
      INFO("temperature " << c.config.temperature << " top_k " << c.config.top_k << " seed "
                          << seed);
      CHECK(std::abs(sum - 1.0) < 1e-5);
      CHECK(support >= 1);
      CHECK(support <= c.max_support);
      if (c.config.temperature == 0.0F) {
        CHECK(support == 1);
      }
      if (c.config.top_p < 1.0F) {
        CHECK(support < static_cast<std::size_t>(kVocab));  // top-p prunes a spread of 97
      }
    }
  }
}

TEST_CASE("a draw from the distribution picks what sample picks with the same generator state",
          "[sampler][distribution]") {
  for (const axon::SamplerConfig& config : {axon::SamplerConfig{.temperature = 0.9F,
                                                                .top_k = 40,
                                                                .top_p = 0.95F,
                                                                .min_p = 0.05F,
                                                                .repeat_penalty = 1.1F,
                                                                .seed = 11},
                                            axon::SamplerConfig{.temperature = 1.5F,
                                                                .top_k = 0,
                                                                .top_p = 1.0F,
                                                                .min_p = 0.0F,
                                                                .repeat_penalty = 1.0F,
                                                                .seed = 12},
                                            axon::SamplerConfig{.temperature = 0.0F, .seed = 13}}) {
    axon::Sampler a(config, kVocab);
    axon::Sampler b(config, kVocab);
    const std::vector<std::int32_t> recent{1, 2, 3};
    std::size_t agreed = 0;
    for (std::uint32_t seed = 100; seed < 300; ++seed) {
      const auto logits = random_logits(seed);
      const std::int32_t direct = a.sample(logits, recent);
      std::vector<float> p(static_cast<std::size_t>(kVocab));
      b.distribution(logits, recent, p);
      const std::int32_t drawn = b.draw(p);
      agreed += direct == drawn ? 1 : 0;
    }
    INFO("temperature " << config.temperature);
    // Equal up to the rounding of the probabilities to float: the two cumulative scans run in
    // the same order over values that differ in the last bits, so a draw landing within that
    // rounding of a boundary can differ. Two hundred rows leave room for a handful.
    CHECK(agreed >= 195);
  }
}

TEST_CASE("draw refuses an empty distribution and honours a point mass",
          "[sampler][distribution]") {
  axon::Sampler sampler({.seed = 5}, kVocab);
  std::vector<float> p(static_cast<std::size_t>(kVocab), 0.0F);
  p[42] = 0.25F;  // any positive total works
  for (int i = 0; i < 10; ++i) {
    CHECK(sampler.draw(p) == 42);
  }
}
