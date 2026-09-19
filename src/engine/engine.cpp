#include "engine/engine.h"

#include "axon/tokenizer.h"

#include <fmt/format.h>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <utility>

namespace axon::engine {

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

double peak_rss_mb() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
#endif
}

std::size_t common_prefix(std::span<const std::int32_t> a, std::span<const std::int32_t> b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

}  // namespace

double Stats::prefill_tok_s() const {
  const std::size_t fed = prompt_tokens - reused_tokens;
  return prefill_s > 0.0 && fed > 0 ? static_cast<double>(fed) / prefill_s : 0.0;
}

double Stats::decode_tok_s() const {
  return decode_s > 0.0 && generated_tokens > 0 ? static_cast<double>(generated_tokens) / decode_s
                                                : 0.0;
}

Engine::Engine(model::LlamaModel& model) : model_(model), cache_(model.new_cache()) {}

void Engine::clear() {
  cache_.clear();
  history_.clear();
}

Expected<std::vector<std::int32_t>>
Engine::generate(std::span<const std::int32_t> prompt, const GenerateOptions& opts,
                 const std::function<void(std::string_view)>& on_text) {
  if (prompt.empty()) {
    return make_error(ErrorCode::InvalidArgument, "generate: empty prompt");
  }
  if (prompt.size() >= cache_.max_ctx()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("generate: prompt has {} tokens, which leaves no room in a "
                                  "context of {}",
                                  prompt.size(), cache_.max_ctx()));
  }
  stats_ = Stats{};
  stats_.prompt_tokens = prompt.size();

  // Reuse the cached prefix. The last prompt token is always re-run because its logits are
  // what the first sample needs, and the cache holds no logits.
  const std::size_t reuse = std::min(common_prefix(history_, prompt), prompt.size() - 1);
  cache_.truncate(reuse);
  history_.resize(reuse);
  stats_.reused_tokens = reuse;

  if (!sampler_ || sampler_->config() != opts.sampler) {
    sampler_.emplace(opts.sampler, static_cast<std::int32_t>(model_.config().n_vocab));
  } else if (opts.sampler.seed != 0) {
    sampler_->reset(opts.sampler.seed);
  }
  stats_.seed = sampler_->seed();

  const Tokenizer& tokenizer = model_.tokenizer();
  const auto stops = opts.stop_tokens.empty() ? tokenizer.eog_tokens() : opts.stop_tokens;
  const auto is_stop = [&](std::int32_t id) { return std::ranges::find(stops, id) != stops.end(); };
  const std::size_t n_vocab = model_.config().n_vocab;
  const auto last_row = [&](std::span<const float> logits) { return logits.last(n_vocab); };

  // Prefill the part of the prompt the cache does not hold, in chunks of max_seq.
  const auto t_prefill = std::chrono::steady_clock::now();
  std::span<const float> logits;
  for (std::size_t offset = reuse; offset < prompt.size(); offset += model_.max_seq()) {
    const std::size_t n = std::min(model_.max_seq(), prompt.size() - offset);
    auto out = model_.forward_prefill(cache_, prompt.subspan(offset, n));
    if (!out) {
      return out.error();
    }
    logits = *out;
    history_.insert(history_.end(), prompt.begin() + static_cast<std::ptrdiff_t>(offset),
                    prompt.begin() + static_cast<std::ptrdiff_t>(offset + n));
  }
  std::int32_t next = sampler_->sample(last_row(logits), history_);
  stats_.prefill_s = seconds_since(t_prefill);

  std::vector<std::int32_t> generated;
  generated.reserve(static_cast<std::size_t>(std::max(opts.max_tokens, 0)));
  StreamingDecoder decoder(tokenizer);
  const auto t_decode = std::chrono::steady_clock::now();
  while (std::cmp_less(generated.size(), opts.max_tokens)) {
    if (is_stop(next)) {
      stats_.stopped_on_token = true;
      break;
    }
    generated.push_back(next);
    const std::string piece = decoder.push(next);
    if (!piece.empty() && on_text) {
      on_text(piece);
    }
    if (std::cmp_greater_equal(generated.size(), opts.max_tokens)) {
      break;
    }
    if (cache_.pos() >= cache_.max_ctx()) {
      stats_.context_full = true;
      break;
    }
    // history_ tracks the cache exactly: a token joins it only when it is fed.
    history_.push_back(next);
    auto out = model_.forward_one(cache_, next);
    if (!out) {
      return out.error();
    }
    next = sampler_->sample(last_row(*out), history_);
  }
  const std::string tail = decoder.flush();
  if (!tail.empty() && on_text) {
    on_text(tail);
  }
  stats_.decode_s = seconds_since(t_decode);
  stats_.generated_tokens = generated.size();
  stats_.peak_rss_mb = peak_rss_mb();
  stats_.cache_positions = cache_.pos();
  stats_.cache_bytes_live = cache_.live_bytes();
  stats_.cache_bytes_reserved = cache_.bytes();
  stats_.arena_bytes = model_.arena_high_water();
  return generated;
}

}  // namespace axon::engine
