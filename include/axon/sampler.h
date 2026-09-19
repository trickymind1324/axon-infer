#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace axon {

struct SamplerConfig {
  float temperature = 0.8F;     // 0 means greedy and skips every other stage
  int top_k = 40;               // 0 disables
  float top_p = 0.95F;          // 1 disables
  float min_p = 0.05F;          // 0 disables
  float repeat_penalty = 1.1F;  // 1 disables
  int repeat_last_n = 64;       // recent tokens the penalty looks at
  std::uint64_t seed = 0;       // 0 draws a seed from the clock, see Sampler::seed()

  bool operator==(const SamplerConfig&) const = default;
};

// Picks the next token from a row of logits. Stages run in the llama.cpp order: repetition
// penalty, temperature, top-k, top-p, min-p, then one draw. All scratch memory is allocated
// at construction, so sampling never touches the heap.
class Sampler {
public:
  Sampler(SamplerConfig config, std::int32_t n_vocab);
  ~Sampler();
  Sampler(Sampler&&) noexcept;
  Sampler& operator=(Sampler&&) noexcept;
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // `recent` is the tail of the sequence, prompt included, for the repetition penalty. The
  // caller's logits are never modified.
  std::int32_t sample(std::span<const float> logits, std::span<const std::int32_t> recent);

  void reset(std::uint64_t seed);
  [[nodiscard]] std::uint64_t seed() const noexcept;
  [[nodiscard]] const SamplerConfig& config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace axon
