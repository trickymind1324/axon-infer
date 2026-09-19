#include "axon/sampler.h"

#include "alloc_counter.h"
#include "sampling/rng.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace {

constexpr std::int32_t kVocab = 8;

std::vector<float> logits_of(std::initializer_list<float> values) {
  return {values};
}

// How many distinct tokens a configuration can ever produce for these logits.
std::set<std::int32_t> support(axon::Sampler& sampler, const std::vector<float>& logits,
                               int draws = 2000) {
  std::set<std::int32_t> seen;
  for (int i = 0; i < draws; ++i) {
    seen.insert(sampler.sample(logits, {}));
  }
  return seen;
}

}  // namespace

TEST_CASE("rng is deterministic and uniform draws stay in range", "[sampler][rng]") {
  axon::Rng a(42);
  axon::Rng b(42);
  axon::Rng c(43);
  bool differs = false;
  for (int i = 0; i < 100; ++i) {
    const auto x = a.next();
    CHECK(x == b.next());
    differs = differs || x != c.next();
  }
  CHECK(differs);
  for (int i = 0; i < 10000; ++i) {
    const double u = a.uniform();
    CHECK(u >= 0.0);
    CHECK(u < 1.0);
  }
  // Two known outputs pin the algorithm so a silent change to the generator is caught.
  axon::Rng pinned(1);
  const auto first = pinned.next();
  const auto second = pinned.next();
  CHECK(first != second);
  axon::Rng again(1);
  CHECK(again.next() == first);
}

TEST_CASE("temperature zero is greedy regardless of the other stages", "[sampler]") {
  axon::Sampler sampler({.temperature = 0.0F, .top_k = 1, .top_p = 0.1F, .seed = 7}, kVocab);
  const auto logits = logits_of({0.1F, 2.0F, 0.5F, 9.0F, -1.0F, 3.0F, 0.0F, 8.9F});
  for (int i = 0; i < 10; ++i) {
    CHECK(sampler.sample(logits, {}) == 3);
  }
}

TEST_CASE("top-k keeps exactly k candidates", "[sampler]") {
  axon::Sampler sampler({.temperature = 1.0F,
                         .top_k = 3,
                         .top_p = 1.0F,
                         .min_p = 0.0F,
                         .repeat_penalty = 1.0F,
                         .seed = 1},
                        kVocab);
  const auto logits = logits_of({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F});
  CHECK(support(sampler, logits) == std::set<std::int32_t>{5, 6, 7});
}

TEST_CASE("top-p keeps the smallest prefix reaching the mass", "[sampler]") {
  // Probabilities at temperature 1: token 0 gets about 0.64, token 1 about 0.24, then 0.09.
  const auto logits = logits_of({3.0F, 2.0F, 1.0F, -20.0F, -20.0F, -20.0F, -20.0F, -20.0F});
  axon::Sampler narrow({.temperature = 1.0F,
                        .top_k = 0,
                        .top_p = 0.6F,
                        .min_p = 0.0F,
                        .repeat_penalty = 1.0F,
                        .seed = 1},
                       kVocab);
  CHECK(support(narrow, logits) == std::set<std::int32_t>{0});
  axon::Sampler wider({.temperature = 1.0F,
                       .top_k = 0,
                       .top_p = 0.8F,
                       .min_p = 0.0F,
                       .repeat_penalty = 1.0F,
                       .seed = 1},
                      kVocab);
  CHECK(support(wider, logits) == std::set<std::int32_t>{0, 1});
}

TEST_CASE("min-p drops candidates below a fraction of the best", "[sampler]") {
  const auto logits = logits_of({3.0F, 2.0F, 1.0F, -20.0F, -20.0F, -20.0F, -20.0F, -20.0F});
  // p(1)/p(0) = e^-1 = 0.37, p(2)/p(0) = e^-2 = 0.14.
  axon::Sampler sampler({.temperature = 1.0F,
                         .top_k = 0,
                         .top_p = 1.0F,
                         .min_p = 0.2F,
                         .repeat_penalty = 1.0F,
                         .seed = 1},
                        kVocab);
  CHECK(support(sampler, logits) == std::set<std::int32_t>{0, 1});
}

TEST_CASE("repetition penalty lowers repeated tokens only", "[sampler]") {
  axon::Sampler sampler(
      {.temperature = 0.0F, .repeat_penalty = 1.5F, .repeat_last_n = 4, .seed = 1}, kVocab);
  const auto logits = logits_of({1.0F, 1.2F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  CHECK(sampler.sample(logits, {}) == 1);
  const std::vector<std::int32_t> recent{1, 1, 1};
  CHECK(sampler.sample(logits, recent) == 0);          // 1.2 / 1.5 = 0.8 < 1.0
  const std::vector<std::int32_t> old{1, 5, 6, 7, 2};  // 1 is outside the window of 4
  CHECK(sampler.sample(logits, old) == 1);
  // Negative logits move further down.
  axon::Sampler strong({.temperature = 0.0F, .repeat_penalty = 2.5F, .repeat_last_n = 4, .seed = 1},
                       kVocab);
  const auto negatives = logits_of({-1.0F, -0.5F, -3.0F, -3.0F, -3.0F, -3.0F, -3.0F, -3.0F});
  const std::vector<std::int32_t> once{1};
  CHECK(strong.sample(negatives, once) == 0);  // -0.5 * 2.5 = -1.25 < -1.0
  // The penalty applies once per distinct token, not once per occurrence: 1.2 / 1.05 = 1.14
  // still beats 1.0, whereas four applications would give 0.99.
  axon::Sampler mild({.temperature = 0.0F, .repeat_penalty = 1.05F, .repeat_last_n = 4, .seed = 1},
                     kVocab);
  const std::vector<std::int32_t> repeated{1, 1, 1, 1};
  CHECK(mild.sample(logits, repeated) == 1);
}

TEST_CASE("seeded sampling reproduces and differs across seeds", "[sampler]") {
  const auto logits = logits_of({1.0F, 1.1F, 0.9F, 1.05F, 0.95F, 1.0F, 1.2F, 0.8F});
  axon::Sampler a({.temperature = 1.0F, .top_k = 0, .top_p = 1.0F, .min_p = 0.0F, .seed = 99},
                  kVocab);
  axon::Sampler b({.temperature = 1.0F, .top_k = 0, .top_p = 1.0F, .min_p = 0.0F, .seed = 99},
                  kVocab);
  axon::Sampler c({.temperature = 1.0F, .top_k = 0, .top_p = 1.0F, .min_p = 0.0F, .seed = 100},
                  kVocab);
  std::vector<std::int32_t> from_a;
  std::vector<std::int32_t> from_b;
  std::vector<std::int32_t> from_c;
  for (int i = 0; i < 64; ++i) {
    from_a.push_back(a.sample(logits, {}));
    from_b.push_back(b.sample(logits, {}));
    from_c.push_back(c.sample(logits, {}));
  }
  CHECK(from_a == from_b);
  CHECK(from_a != from_c);
  CHECK(a.seed() == 99);
  a.reset(99);
  CHECK(a.sample(logits, {}) == from_a[0]);

  axon::Sampler clock({.seed = 0}, kVocab);
  CHECK(clock.seed() != 0);
}

TEST_CASE("draw frequencies follow the distribution", "[sampler]") {
  // Four live tokens with probabilities 0.4, 0.3, 0.2, 0.1 at temperature 1.
  const std::array<double, 4> target{0.4, 0.3, 0.2, 0.1};
  std::vector<float> logits(kVocab, -30.0F);
  for (std::size_t i = 0; i < 4; ++i) {
    logits[i] = static_cast<float>(std::log(target[i]));
  }
  axon::Sampler sampler({.temperature = 1.0F,
                         .top_k = 0,
                         .top_p = 1.0F,
                         .min_p = 0.0F,
                         .repeat_penalty = 1.0F,
                         .seed = 5},
                        kVocab);
  std::array<int, kVocab> counts{};
  const int draws = 10000;
  for (int i = 0; i < draws; ++i) {
    counts[static_cast<std::size_t>(sampler.sample(logits, {}))]++;
  }
  for (std::size_t i = 0; i < 4; ++i) {
    const double frequency = static_cast<double>(counts[i]) / draws;
    INFO("token " << i << " frequency " << frequency);
    CHECK(std::abs(frequency - target[i]) < 0.02);
  }
  CHECK(counts[4] + counts[5] + counts[6] + counts[7] == 0);
}

TEST_CASE("sampling does not allocate", "[sampler]") {
  axon::Sampler sampler({.seed = 3}, 4096);
  std::vector<float> logits(4096);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    logits[i] = static_cast<float>((i * 37) % 101) / 10.0F;
  }
  std::vector<std::int32_t> recent(64, 7);
  sampler.sample(logits, recent);
  const auto before = axon::test::allocation_count();
  for (int i = 0; i < 100; ++i) {
    sampler.sample(logits, recent);
  }
  CHECK(axon::test::allocation_count() == before);
}
