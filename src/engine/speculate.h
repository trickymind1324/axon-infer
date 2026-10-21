#pragma once

// Speculative decoding: a draft model proposes a few tokens, the target scores them all in one
// multi-row pass, and the longest prefix the target agrees with is kept. The target's
// multi-row pass is bit-identical to its single-row decode, so with greedy sampling the
// output equals the target decoding alone, token for token. With sampling, the acceptance
// rule keeps the target's distribution exactly.

#include "axon/sampler.h"
#include "core/expected.h"

#include "engine/sequence.h"
#include "model/kv_cache.h"
#include "model/llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace axon::engine {

struct SpeculateOptions {
  std::size_t draft_tokens = 4;  // tokens the draft proposes per round
};

struct SpeculateStats {
  std::uint64_t rounds = 0;
  std::uint64_t drafted = 0;   // tokens the draft proposed
  std::uint64_t accepted = 0;  // of those, kept
  std::uint64_t emitted = 0;   // tokens the target produced, accepted plus one per round
  [[nodiscard]] double acceptance() const noexcept {
    return drafted > 0 ? static_cast<double>(accepted) / static_cast<double>(drafted) : 0.0;
  }
  [[nodiscard]] double tokens_per_pass() const noexcept {
    return rounds > 0 ? static_cast<double>(emitted) / static_cast<double>(rounds) : 0.0;
  }
};

// The acceptance rule for one drafted token. p is the target's distribution and q the
// draft's at the same position, both as the sampler would draw from them. Returns the token
// to emit: `drafted` when accepted (with probability min(1, p / q) at it), otherwise a draw
// from the residual max(0, p - q), which keeps the emitted token distributed as p. The
// residual uses `scratch` of the vocabulary size.
std::int32_t accept_or_resample(std::span<const float> p, std::span<const float> q,
                                std::int32_t drafted, Sampler& sampler, std::span<float> scratch,
                                bool* accepted);

class Speculator {
public:
  // The draft must share the target's vocabulary. Fails otherwise.
  static Expected<bool> compatible(const model::LlamaModel& draft, const model::LlamaModel& target);
  Speculator(model::LlamaModel& draft, const model::LlamaModel& target, SpeculateOptions opts);

  // One round on a sequence in decode: its pending token goes through the end rules and is
  // fed, the draft proposes up to K tokens from there, the target scores them in one pass,
  // and the accepted prefix plus one target-sampled token follow in order through the same
  // end rules. Afterwards the caches and the history hold the accepted tokens and the
  // sequence's pending token is the next to feed, or the sequence is done. Returns how many
  // tokens the sequence accepted this round (accepted drafts plus the one sampled), zero when
  // the pending token ended the generation.
  Expected<std::size_t> round(Sequence& seq);

  [[nodiscard]] const SpeculateStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::size_t draft_tokens() const noexcept { return opts_.draft_tokens; }

private:
  // Brings the draft cache to the sequence's history, reusing the common prefix.
  Expected<bool> sync(std::span<const std::int32_t> history);

  model::LlamaModel& draft_;
  SpeculateOptions opts_;
  model::KVCache draft_cache_;
  std::vector<std::int32_t> draft_history_;  // what the draft cache holds
  SamplerConfig draft_config_;
  std::unique_ptr<Sampler> draft_sampler_;
  std::vector<std::int32_t> proposed_;
  std::vector<float> q_;  // K distributions of the draft, n_vocab each
  std::vector<float> p_;  // one target distribution
  std::vector<float> residual_;
  std::vector<std::int32_t> fed_;  // the target pass: pending token plus the drafts
  SpeculateStats stats_;
};

}  // namespace axon::engine
