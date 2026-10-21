#include "engine/speculate.h"

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace axon::engine {

std::int32_t accept_or_resample(std::span<const float> p, std::span<const float> q,
                                std::int32_t drafted, Sampler& sampler, std::span<float> scratch,
                                bool* accepted) {
  const auto d = static_cast<std::size_t>(drafted);
  const auto pd = static_cast<double>(p[d]);
  const auto qd = static_cast<double>(q[d]);
  // The draft drew `drafted`, so q at it is positive. Accept with probability min(1, p / q).
  const double u = sampler.uniform();
  if (qd > 0.0 && u * qd < pd) {
    *accepted = true;
    return drafted;
  }
  *accepted = false;
  double total = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const float r = p[i] - q[i];
    scratch[i] = r > 0.0F ? r : 0.0F;
    total += static_cast<double>(scratch[i]);
  }
  if (total <= 0.0) {
    // p and q agree everywhere p lives, which can only happen when they are equal and the
    // test above accepted. Rounding can get here, and then p itself is the right answer.
    return sampler.draw(p);
  }
  return sampler.draw(scratch);
}

Expected<bool> Speculator::compatible(const model::LlamaModel& draft,
                                      const model::LlamaModel& target) {
  if (draft.config().n_vocab != target.config().n_vocab) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("speculate: the draft has a vocabulary of {} tokens and the "
                                  "target {}",
                                  draft.config().n_vocab, target.config().n_vocab));
  }
  const auto& a = draft.tokenizer();
  const auto& b = target.tokenizer();
  if (a.vocab_size() != b.vocab_size() || a.bos() != b.bos()) {
    return make_error(ErrorCode::InvalidArgument,
                      "speculate: the draft and target tokenizers differ");
  }
  // A sample of pieces across the vocabulary, so two tokenizers with the same size but
  // different tables are caught without comparing every entry at load.
  const std::int32_t n = a.vocab_size();
  for (std::int32_t id = 0; id < n; id += std::max(1, n / 256)) {
    if (a.token_text(id) != b.token_text(id)) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("speculate: token {} differs between the draft and the "
                                    "target tokenizers",
                                    id));
    }
  }
  if (draft.max_ctx() < target.max_ctx()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("speculate: the draft's context of {} is shorter than the "
                                  "target's {}",
                                  draft.max_ctx(), target.max_ctx()));
  }
  if (draft.max_seq() < target.max_seq()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("speculate: the draft's max_seq of {} is below the target's {}",
                                  draft.max_seq(), target.max_seq()));
  }
  return true;
}

Speculator::Speculator(model::LlamaModel& draft, const model::LlamaModel& target,
                       SpeculateOptions opts)
    : draft_(draft), opts_(opts), draft_cache_(draft.new_cache()) {
  opts_.draft_tokens = std::max<std::size_t>(1, std::min(opts_.draft_tokens, target.max_seq() - 1));
  const auto n_vocab = static_cast<std::size_t>(target.config().n_vocab);
  q_.resize(opts_.draft_tokens * n_vocab);
  p_.resize(n_vocab);
  residual_.resize(n_vocab);
  proposed_.reserve(opts_.draft_tokens);
  fed_.reserve(opts_.draft_tokens + 1);
}

Expected<bool> Speculator::sync(std::span<const std::int32_t> history) {
  std::size_t shared = 0;
  const std::size_t n = std::min(history.size(), draft_history_.size());
  while (shared < n && history[shared] == draft_history_[shared]) {
    ++shared;
  }
  draft_cache_.truncate(shared);
  draft_history_.resize(shared);
  // Everything the target holds beyond the shared prefix is prefilled into the draft, in
  // chunks of its max_seq. The last target token is fed too, since the draft continues from
  // the same position the target is at.
  for (std::size_t offset = shared; offset < history.size(); offset += draft_.max_seq()) {
    const std::size_t count = std::min(draft_.max_seq(), history.size() - offset);
    const auto out = draft_.forward_prefill(draft_cache_, history.subspan(offset, count));
    if (!out) {
      return out.error();
    }
    draft_history_.insert(draft_history_.end(),
                          history.begin() + static_cast<std::ptrdiff_t>(offset),
                          history.begin() + static_cast<std::ptrdiff_t>(offset + count));
  }
  return true;
}

Expected<std::size_t> Speculator::round(Sequence& seq) {
  if (!seq.accept_pending()) {
    return std::size_t{0};
  }
  const std::int32_t first = seq.pending();
  seq.fed();
  // The draft and the target each need their own sampler state, with the same stages.
  if (!draft_sampler_ || draft_config_ != seq.sampler().config()) {
    draft_config_ = seq.sampler().config();
    SamplerConfig config = draft_config_;
    // Its own generator stream: a fixed seed for the target still gives the draft a different
    // one, so the acceptance draws and the draft draws do not share a sequence.
    config.seed = config.seed == 0 ? 0 : config.seed ^ 0x9E3779B97F4A7C15ULL;
    draft_sampler_ = std::make_unique<Sampler>(config, draft_.config().n_vocab);
  }
  const auto n_vocab = static_cast<std::size_t>(draft_.config().n_vocab);
  const std::size_t room = seq.cache().max_ctx() - seq.cache().pos();
  // The target pass feeds the pending token plus the drafts, and the cache must hold all of
  // them, so near the end of the context the draft shortens.
  const std::size_t k = std::min(opts_.draft_tokens, room > 1 ? room - 1 : 0);

  // The draft: brought to the target's history up to the token just fed, which it is then
  // fed on its own, then K steps, each sampled from the draft's own distribution.
  const auto synced = sync(seq.history().first(seq.history().size() - 1));
  if (!synced) {
    return synced.error();
  }
  proposed_.clear();
  std::span<const float> draft_logits;
  {
    // The history's last token is the one just fed to the target and not yet to the draft.
    const auto out = draft_.forward_one(draft_cache_, first);
    if (!out) {
      return out.error();
    }
    draft_history_.push_back(first);
    draft_logits = out->last(n_vocab);
  }
  for (std::size_t i = 0; i < k; ++i) {
    std::span<float> q(q_.data() + i * n_vocab, n_vocab);
    draft_sampler_->distribution(draft_logits, draft_history_, q);
    const std::int32_t token = draft_sampler_->draw(q);
    proposed_.push_back(token);
    if (i + 1 < k) {
      const auto out = draft_.forward_one(draft_cache_, token);
      if (!out) {
        return out.error();
      }
      draft_history_.push_back(token);
      draft_logits = out->last(n_vocab);
    }
  }

  // The target: one pass over the fed token and the drafts, K + 1 rows of logits.
  fed_.clear();
  fed_.push_back(first);
  fed_.insert(fed_.end(), proposed_.begin(), proposed_.end());
  const auto rows = seq.model().forward_prefill(seq.cache(), fed_);
  if (!rows) {
    return rows.error();
  }

  // Acceptance, in order, each kept token going through the end rules as a plain decode
  // would run them.
  std::size_t kept = 0;
  std::size_t emitted = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const auto row = rows->subspan(i * n_vocab, n_vocab);
    seq.sampler().distribution(row, seq.history(), p_);
    const std::span<const float> q(q_.data() + i * n_vocab, n_vocab);
    bool accepted = false;
    const std::int32_t token =
        accept_or_resample(p_, q, proposed_[i], seq.sampler(), residual_, &accepted);
    if (accepted) {
      ++kept;
    }
    seq.set_pending(token);
    ++emitted;
    if (!accepted) {
      break;  // the resampled token is the new pending token, the rest of the draft is dropped
    }
    if (!seq.accept_pending()) {
      break;  // a stop token, the length or the cancel flag ended the generation here
    }
    seq.fed();
    if (i + 1 == k) {
      // Every draft accepted: the last row gives one more token from the target.
      const auto last = rows->subspan(k * n_vocab, n_vocab);
      seq.sampler().distribution(last, seq.history(), p_);
      seq.set_pending(seq.sampler().draw(p_));
      ++emitted;
    }
  }
  if (k == 0) {
    // No room to draft: the pass scored only the fed token.
    seq.sampler().distribution(rows->subspan(0, n_vocab), seq.history(), p_);
    seq.set_pending(seq.sampler().draw(p_));
    ++emitted;
  }
  // The target cache holds every token of the pass. Keep what the history holds.
  seq.cache().truncate(seq.history().size());
  stats_.rounds += 1;
  stats_.drafted += k;
  stats_.accepted += kept;
  stats_.emitted += emitted;
  return emitted;
}

}  // namespace axon::engine
