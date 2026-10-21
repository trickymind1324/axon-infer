#pragma once

// One generation in progress: a cache, the tokens it holds, a sampler, a streaming decoder and
// the rules that end a generation. The single-sequence Engine drives one of these through
// forward_one, the batch engine drives several through one forward_batch per step, and both
// produce the same text for the same prompt because the rules live here once.

#include "axon/sampler.h"
#include "axon/tokenizer.h"
#include "core/expected.h"

#include "engine/options.h"
#include "model/kv_cache.h"
#include "model/llama.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace axon::engine {

enum class Finish : std::uint8_t {
  Stop,       // a stop token
  Length,     // max_tokens reached, or the context filled up
  Cancelled,  // the cancel flag, which a server also uses for a completed stop string
};

class Sequence {
public:
  explicit Sequence(model::LlamaModel& model);

  enum class State : std::uint8_t { Idle, Prefill, Decode, Done };
  [[nodiscard]] State state() const noexcept { return state_; }
  [[nodiscard]] bool idle() const noexcept {
    return state_ == State::Idle || state_ == State::Done;
  }

  // Begins a generation. The cache keeps the longest prefix it shares with the prompt and
  // only the rest is prefilled. Fails when the prompt is empty or does not fit the context.
  // The stop tokens of `opts` are copied, the cancel flag is kept by pointer.
  Expected<bool> start(std::span<const std::int32_t> prompt, const GenerateOptions& opts,
                       std::function<void(std::string_view)> on_text);
  // Prefill: feeds the next chunk of the prompt, at most max_seq tokens. When the prompt is
  // complete the first token is sampled and the state moves to Decode. True when complete.
  Expected<bool> prefill_chunk();
  // Decode: applies the end rules to the sampled token. True when it was accepted and must be
  // fed, false when the generation ended (the text is flushed and the state is Done).
  bool accept_pending();
  [[nodiscard]] std::int32_t pending() const noexcept { return next_; }
  // Decode: replaces the pending token, for a speculator that chose it from the target's
  // distribution instead of observe().
  void set_pending(std::int32_t token) noexcept { next_ = token; }
  [[nodiscard]] model::KVCache& cache() noexcept { return cache_; }
  [[nodiscard]] model::LlamaModel& model() noexcept { return model_; }
  // The sampler of the generation in progress, valid from start() on.
  [[nodiscard]] Sampler& sampler() noexcept { return *sampler_; }
  // Decode: the accepted token was fed and the cache holds it. Then observe() the logits.
  void fed();
  void observe(std::span<const float> logits_row);

  [[nodiscard]] Finish finish() const noexcept { return finish_; }
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::span<const std::int32_t> history() const noexcept { return history_; }
  [[nodiscard]] std::span<const std::int32_t> generated() const noexcept { return generated_; }
  [[nodiscard]] std::size_t prompt_remaining() const noexcept { return prompt_.size() - fed_; }
  // Drops the oldest tokens after the first `keep`, half of them, and re-prefills what stays,
  // so the cache has room again. Exact, since nothing cached is approximated. False when
  // there is nothing to drop.
  Expected<bool> shift(std::size_t keep);
  void clear();  // forgets the cache and history
  // Ends a generation without the usual flush, for a pass that failed underneath it.
  void abandon() noexcept { state_ = State::Done; }
  // How many leading tokens of `prompt` the cache already holds.
  [[nodiscard]] std::size_t shared_prefix(std::span<const std::int32_t> prompt) const noexcept;

private:
  void end(Finish finish);
  [[nodiscard]] bool is_stop(std::int32_t id) const;

  model::LlamaModel& model_;
  model::KVCache cache_;
  std::vector<std::int32_t> history_;  // every token the cache holds, in order
  std::vector<std::int32_t> generated_;
  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<StreamingDecoder> decoder_;
  std::function<void(std::string_view)> on_text_;
  std::vector<std::int32_t> prompt_;
  std::vector<std::int32_t> stop_tokens_;
  GenerateOptions opts_;
  std::size_t fed_ = 0;  // prompt tokens the cache holds for this generation
  std::int32_t next_ = 0;
  State state_ = State::Idle;
  Finish finish_ = Finish::Stop;
  Stats stats_;
  std::chrono::steady_clock::time_point t_start_;
  std::chrono::steady_clock::time_point t_decode_;
};

}  // namespace axon::engine
