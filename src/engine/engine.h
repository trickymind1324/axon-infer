#pragma once

// The generation loop: feeds a prompt through the KV cache, reusing whatever prefix the cache
// already holds, then samples tokens one at a time and streams the decoded text.

#include "axon/sampler.h"
#include "core/expected.h"

#include "model/kv_cache.h"
#include "model/llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace axon::engine {

struct GenerateOptions {
  int max_tokens = 256;
  SamplerConfig sampler;
  // Ends generation when sampled. Empty means the tokenizer's end-of-generation tokens.
  std::span<const std::int32_t> stop_tokens;
};

struct Stats {
  std::size_t prompt_tokens = 0;  // tokens in the prompt
  std::size_t reused_tokens = 0;  // of those, how many the cache already held
  std::size_t generated_tokens = 0;
  double prefill_s = 0.0;  // time to first token
  double decode_s = 0.0;
  double peak_rss_mb = 0.0;
  std::size_t cache_positions = 0;
  std::size_t cache_bytes_live = 0;
  std::size_t cache_bytes_reserved = 0;
  std::size_t arena_bytes = 0;
  std::uint64_t seed = 0;
  bool stopped_on_token = false;  // a stop token ended generation
  bool context_full = false;      // the cache filled up before max_tokens
  [[nodiscard]] double prefill_tok_s() const;
  [[nodiscard]] double decode_tok_s() const;
};

class Engine {
public:
  explicit Engine(model::LlamaModel& model);

  // Runs the prompt, then samples up to max_tokens, calling on_text with each decoded piece
  // as soon as it is complete. Returns the generated ids. A stop token or a full context ends
  // generation early and is reported in stats(). Fails only when the prompt itself cannot be
  // processed, for example because it does not fit the context.
  Expected<std::vector<std::int32_t>>
  generate(std::span<const std::int32_t> prompt, const GenerateOptions& opts,
           const std::function<void(std::string_view)>& on_text);

  void clear();  // forgets the cache and history
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::span<const std::int32_t> history() const noexcept { return history_; }
  [[nodiscard]] model::LlamaModel& model() noexcept { return model_; }

private:
  model::LlamaModel& model_;
  model::KVCache cache_;
  std::vector<std::int32_t> history_;  // every token the cache holds, in order
  std::optional<Sampler> sampler_;
  Stats stats_;
};

}  // namespace axon::engine
