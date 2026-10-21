#pragma once

// What a generation takes and what it reports, shared by the single-sequence engine and the
// batch engine.

#include "axon/sampler.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

namespace axon::engine {

struct GenerateOptions {
  int max_tokens = 256;
  SamplerConfig sampler;
  // Ends generation when sampled. Empty means the tokenizer's end-of-generation tokens.
  std::span<const std::int32_t> stop_tokens;
  // False ignores stop tokens entirely and always produces max_tokens, for benchmarks.
  bool stop_on_tokens = true;
  // Checked before every decode step. Set from another thread to end generation after the
  // token in flight, which is how a server stops a request whose client has gone.
  const std::atomic<bool>* cancel = nullptr;
  // When the cache fills, keep the first `keep` tokens (a system prompt), drop the oldest
  // half of the rest, re-prefill what remains and carry on, instead of ending with length.
  bool context_shift = false;
  std::size_t keep = 0;
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
  bool stopped_on_token = false;   // a stop token ended generation
  bool context_full = false;       // the cache filled up before max_tokens
  bool cancelled = false;          // the cancel flag ended generation
  std::size_t context_shifts = 0;  // times the cache was shifted during this generation
  std::size_t last_shift_at = 0;   // generated tokens at the last shift, the pending one included
  [[nodiscard]] double prefill_tok_s() const;
  [[nodiscard]] double decode_tok_s() const;
};

}  // namespace axon::engine
