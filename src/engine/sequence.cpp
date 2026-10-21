#include "engine/sequence.h"

#include <fmt/format.h>

#include <sys/resource.h>

#include <algorithm>
#include <memory>
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

Sequence::Sequence(model::LlamaModel& model) : model_(model), cache_(model.new_cache()) {}

std::size_t Sequence::shared_prefix(std::span<const std::int32_t> prompt) const noexcept {
  return common_prefix(history_, prompt);
}

void Sequence::clear() {
  cache_.clear();
  history_.clear();
  generated_.clear();
  state_ = State::Idle;
}

Expected<bool> Sequence::start(std::span<const std::int32_t> prompt, const GenerateOptions& opts,
                               std::function<void(std::string_view)> on_text) {
  if (prompt.empty()) {
    return make_error(ErrorCode::InvalidArgument, "generate: empty prompt");
  }
  if (prompt.size() >= cache_.max_ctx()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("generate: prompt has {} tokens, which leaves no room in a "
                                  "context of {}",
                                  prompt.size(), cache_.max_ctx()));
  }
  opts_ = opts;
  stop_tokens_.assign(opts.stop_tokens.begin(), opts.stop_tokens.end());
  if (stop_tokens_.empty()) {
    const auto eog = model_.tokenizer().eog_tokens();
    stop_tokens_.assign(eog.begin(), eog.end());
  }
  on_text_ = std::move(on_text);
  prompt_.assign(prompt.begin(), prompt.end());
  generated_.clear();
  stats_ = Stats{};
  stats_.prompt_tokens = prompt.size();
  // Reuse the cached prefix. The last prompt token is always re-run because its logits are
  // what the first sample needs, and the cache holds no logits.
  const std::size_t reuse = std::min(common_prefix(history_, prompt), prompt.size() - 1);
  cache_.truncate(reuse);
  history_.resize(reuse);
  fed_ = reuse;
  stats_.reused_tokens = reuse;
  if (!sampler_ || sampler_->config() != opts.sampler) {
    sampler_ =
        std::make_unique<Sampler>(opts.sampler, static_cast<std::int32_t>(model_.config().n_vocab));
  } else if (opts.sampler.seed != 0) {
    sampler_->reset(opts.sampler.seed);
  }
  stats_.seed = sampler_->seed();
  if (!decoder_) {
    decoder_ = std::make_unique<StreamingDecoder>(model_.tokenizer());
  } else {
    decoder_->reset();
  }
  t_start_ = std::chrono::steady_clock::now();
  state_ = State::Prefill;
  return true;
}

Expected<bool> Sequence::prefill_chunk() {
  if (state_ != State::Prefill) {
    return make_error(ErrorCode::InvalidArgument, "prefill: the sequence is not prefilling");
  }
  const std::size_t n = std::min(model_.max_seq(), prompt_.size() - fed_);
  const auto out = model_.forward_prefill(cache_, std::span(prompt_).subspan(fed_, n));
  if (!out) {
    state_ = State::Done;
    return out.error();
  }
  history_.insert(history_.end(), prompt_.begin() + static_cast<std::ptrdiff_t>(fed_),
                  prompt_.begin() + static_cast<std::ptrdiff_t>(fed_ + n));
  fed_ += n;
  if (fed_ < prompt_.size()) {
    return false;
  }
  const std::size_t n_vocab = model_.config().n_vocab;
  next_ = sampler_->sample(out->last(n_vocab), history_);
  stats_.prefill_s = seconds_since(t_start_);
  t_decode_ = std::chrono::steady_clock::now();
  state_ = State::Decode;
  return true;
}

bool Sequence::is_stop(std::int32_t id) const {
  return opts_.stop_on_tokens && std::ranges::find(stop_tokens_, id) != stop_tokens_.end();
}

bool Sequence::accept_pending() {
  if (state_ != State::Decode) {
    return false;
  }
  if (std::cmp_greater_equal(generated_.size(), opts_.max_tokens)) {
    end(Finish::Length);
    return false;
  }
  if (is_stop(next_)) {
    stats_.stopped_on_token = true;
    end(Finish::Stop);
    return false;
  }
  generated_.push_back(next_);
  const std::string piece = decoder_->push(next_);
  if (!piece.empty() && on_text_) {
    on_text_(piece);
  }
  if (std::cmp_greater_equal(generated_.size(), opts_.max_tokens)) {
    end(Finish::Length);
    return false;
  }
  // Full means the fed history fills the context. The cache itself may hold more during a
  // speculative round, where drafts not yet accepted sit in it, and those do not count.
  if (history_.size() >= cache_.max_ctx()) {
    bool shifted = false;
    if (opts_.context_shift) {
      const auto done = shift(opts_.keep);
      shifted = done && *done;
    }
    if (!shifted) {
      stats_.context_full = true;
      end(Finish::Length);
      return false;
    }
    ++stats_.context_shifts;
    stats_.last_shift_at = generated_.size();
  }
  if (opts_.cancel != nullptr && opts_.cancel->load(std::memory_order_relaxed)) {
    stats_.cancelled = true;
    end(Finish::Cancelled);
    return false;
  }
  return true;
}

Expected<bool> Sequence::shift(std::size_t keep) {
  keep = std::min(keep, history_.size());
  const std::size_t rest = history_.size() - keep;
  const std::size_t drop = rest / 2;
  if (drop == 0) {
    return false;
  }
  history_.erase(history_.begin() + static_cast<std::ptrdiff_t>(keep),
                 history_.begin() + static_cast<std::ptrdiff_t>(keep + drop));
  cache_.truncate(keep);
  for (std::size_t offset = keep; offset < history_.size(); offset += model_.max_seq()) {
    const std::size_t n = std::min(model_.max_seq(), history_.size() - offset);
    const auto out = model_.forward_prefill(cache_, std::span(history_).subspan(offset, n));
    if (!out) {
      return out.error();
    }
  }
  return true;
}

void Sequence::fed() {
  // history_ tracks the cache exactly: a token joins it only when it has been fed.
  history_.push_back(next_);
}

void Sequence::observe(std::span<const float> logits_row) {
  next_ = sampler_->sample(logits_row, history_);
}

void Sequence::end(Finish finish) {
  finish_ = finish;
  const std::string tail = decoder_->flush();
  if (!tail.empty() && on_text_) {
    on_text_(tail);
  }
  stats_.decode_s = seconds_since(t_decode_);
  stats_.generated_tokens = generated_.size();
  stats_.peak_rss_mb = peak_rss_mb();
  stats_.cache_positions = cache_.pos();
  stats_.cache_bytes_live = cache_.live_bytes();
  stats_.cache_bytes_reserved = cache_.bytes();
  stats_.arena_bytes = model_.arena_high_water();
  state_ = State::Done;
}

}  // namespace axon::engine
