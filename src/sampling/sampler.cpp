#include "axon/sampler.h"

#include "core/check.h"

#include "sampling/rng.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace axon {

struct Sampler::Impl {
  SamplerConfig config;
  std::int32_t n_vocab;
  std::uint64_t seed = 0;
  Rng rng;
  std::vector<float> logits;          // working copy
  std::vector<std::int32_t> order;    // candidate ids, sorted by logit descending
  std::vector<double> probs;          // probabilities of the candidates, same order
  std::vector<std::uint8_t> touched;  // per-id flag so the penalty applies once per token
};

namespace {

std::uint64_t clock_seed() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return static_cast<std::uint64_t>(now) ^ 0xD1B54A32D192ED03ULL;
}

}  // namespace

Sampler::Sampler(SamplerConfig config, std::int32_t n_vocab) : impl_(std::make_unique<Impl>()) {
  AXON_CHECK(n_vocab > 0, "Sampler: vocabulary size {}", n_vocab);
  impl_->config = config;
  impl_->n_vocab = n_vocab;
  const auto n = static_cast<std::size_t>(n_vocab);
  impl_->logits.resize(n);
  impl_->order.resize(n);
  impl_->probs.resize(n);
  impl_->touched.assign(n, 0);
  reset(config.seed);
}

Sampler::~Sampler() = default;
Sampler::Sampler(Sampler&&) noexcept = default;
Sampler& Sampler::operator=(Sampler&&) noexcept = default;

void Sampler::reset(std::uint64_t seed) {
  impl_->seed = seed == 0 ? clock_seed() : seed;
  impl_->rng.seed(impl_->seed);
}

std::uint64_t Sampler::seed() const noexcept {
  return impl_->seed;
}

const SamplerConfig& Sampler::config() const noexcept {
  return impl_->config;
}

// Runs every stage but the draw. Leaves the kept candidates in s.order[0, kept) in descending
// order with their probabilities in s.probs, unnormalised over the kept set. With temperature
// zero returns the argmax in `greedy` and kept zero.
std::size_t Sampler::prepare(std::span<const float> logits, std::span<const std::int32_t> recent,
                             std::int32_t& greedy) {
  Impl& s = *impl_;
  const SamplerConfig& c = s.config;
  const auto n = static_cast<std::size_t>(s.n_vocab);
  AXON_CHECK(logits.size() == n, "Sampler: expected {} logits, found {}", n, logits.size());
  std::ranges::copy(logits, s.logits.begin());

  // Repetition penalty over the last repeat_last_n tokens, once per distinct token.
  if (c.repeat_penalty != 1.0F && c.repeat_last_n > 0 && !recent.empty()) {
    const std::size_t window = std::min(recent.size(), static_cast<std::size_t>(c.repeat_last_n));
    const auto tail = recent.last(window);
    for (const std::int32_t id : tail) {
      if (id < 0 || std::cmp_greater_equal(id, n)) {
        continue;
      }
      const auto i = static_cast<std::size_t>(id);
      if (s.touched[i] != 0) {
        continue;
      }
      s.touched[i] = 1;
      float& logit = s.logits[i];
      logit = logit > 0.0F ? logit / c.repeat_penalty : logit * c.repeat_penalty;
    }
    for (const std::int32_t id : tail) {
      if (id >= 0 && std::cmp_less(id, n)) {
        s.touched[static_cast<std::size_t>(id)] = 0;
      }
    }
  }

  if (c.temperature <= 0.0F) {
    greedy = static_cast<std::int32_t>(std::ranges::max_element(s.logits) - s.logits.begin());
    return 0;
  }

  // Candidates sorted by logit, descending. Top-k needs only the first k in order.
  for (std::size_t i = 0; i < n; ++i) {
    s.order[i] = static_cast<std::int32_t>(i);
  }
  const auto by_logit = [&](std::int32_t a, std::int32_t b) {
    return s.logits[static_cast<std::size_t>(a)] > s.logits[static_cast<std::size_t>(b)];
  };
  std::size_t kept = n;
  if (c.top_k > 0 && std::cmp_less(c.top_k, n)) {
    kept = static_cast<std::size_t>(c.top_k);
    std::ranges::partial_sort(s.order, s.order.begin() + static_cast<std::ptrdiff_t>(kept),
                              by_logit);
  } else {
    std::ranges::sort(s.order, by_logit);
  }

  // Temperature and softmax over the kept candidates, in double for the cumulative sums.
  const auto max_logit = static_cast<double>(s.logits[static_cast<std::size_t>(s.order[0])]);
  const auto temperature = static_cast<double>(c.temperature);
  double sum = 0.0;
  for (std::size_t i = 0; i < kept; ++i) {
    const auto logit = static_cast<double>(s.logits[static_cast<std::size_t>(s.order[i])]);
    s.probs[i] = std::exp((logit - max_logit) / temperature);
    sum += s.probs[i];
  }
  for (std::size_t i = 0; i < kept; ++i) {
    s.probs[i] /= sum;
  }

  // Top-p: the shortest prefix whose mass reaches p.
  if (c.top_p < 1.0F) {
    double cumulative = 0.0;
    std::size_t cut = kept;
    for (std::size_t i = 0; i < kept; ++i) {
      cumulative += s.probs[i];
      if (cumulative >= static_cast<double>(c.top_p)) {
        cut = i + 1;
        break;
      }
    }
    kept = cut;
  }

  // Min-p: drop everything below a fraction of the best candidate.
  if (c.min_p > 0.0F) {
    const double floor = static_cast<double>(c.min_p) * s.probs[0];
    std::size_t cut = kept;
    for (std::size_t i = 1; i < kept; ++i) {
      if (s.probs[i] < floor) {
        cut = i;
        break;
      }
    }
    kept = cut;
  }
  return kept;
}

std::int32_t Sampler::sample(std::span<const float> logits, std::span<const std::int32_t> recent) {
  Impl& s = *impl_;
  std::int32_t greedy = 0;
  const std::size_t kept = prepare(logits, recent, greedy);
  if (kept == 0) {
    return greedy;
  }
  double remaining = 0.0;
  for (std::size_t i = 0; i < kept; ++i) {
    remaining += s.probs[i];
  }
  const double u = s.rng.uniform() * remaining;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < kept; ++i) {
    cumulative += s.probs[i];
    if (u < cumulative) {
      return s.order[i];
    }
  }
  return s.order[kept - 1];  // rounding at the very top of the range
}

void Sampler::distribution(std::span<const float> logits, std::span<const std::int32_t> recent,
                           std::span<float> out) {
  Impl& s = *impl_;
  const auto n = static_cast<std::size_t>(s.n_vocab);
  AXON_CHECK(out.size() == n, "Sampler: expected room for {} probabilities, found {}", n,
             out.size());
  std::ranges::fill(out, 0.0F);
  std::int32_t greedy = 0;
  const std::size_t kept = prepare(logits, recent, greedy);
  if (kept == 0) {
    out[static_cast<std::size_t>(greedy)] = 1.0F;
    return;
  }
  double remaining = 0.0;
  for (std::size_t i = 0; i < kept; ++i) {
    remaining += s.probs[i];
  }
  for (std::size_t i = 0; i < kept; ++i) {
    out[static_cast<std::size_t>(s.order[i])] = static_cast<float>(s.probs[i] / remaining);
  }
}

double Sampler::uniform() {
  return impl_->rng.uniform();
}

std::int32_t Sampler::draw(std::span<const float> probabilities) {
  Impl& s = *impl_;
  const auto n = static_cast<std::size_t>(s.n_vocab);
  AXON_CHECK(probabilities.size() == n, "Sampler: expected {} probabilities, found {}", n,
             probabilities.size());
  // The support, largest first, so the scan order matches sample()'s.
  std::size_t support = 0;
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (probabilities[i] > 0.0F) {
      s.order[support++] = static_cast<std::int32_t>(i);
      total += static_cast<double>(probabilities[i]);
    }
  }
  AXON_CHECK(support > 0 && total > 0.0, "Sampler: a draw from an empty distribution");
  std::sort(s.order.begin(), s.order.begin() + static_cast<std::ptrdiff_t>(support),
            [&](std::int32_t a, std::int32_t b) {
              return probabilities[static_cast<std::size_t>(a)] >
                     probabilities[static_cast<std::size_t>(b)];
            });
  const double u = s.rng.uniform() * total;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < support; ++i) {
    cumulative += static_cast<double>(probabilities[static_cast<std::size_t>(s.order[i])]);
    if (u < cumulative) {
      return s.order[i];
    }
  }
  return s.order[support - 1];
}

}  // namespace axon
