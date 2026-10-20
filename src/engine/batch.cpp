#include "engine/batch.h"

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace axon::engine {

BatchEngine::BatchEngine(model::LlamaModel& model, std::size_t slots) : model_(model) {
  const std::size_t n = std::max<std::size_t>(1, std::min(slots, model.max_batch()));
  sequences_.reserve(n);
  slots_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    sequences_.push_back(std::make_unique<Sequence>(model));
    slots_.push_back(Slot{.done = {}, .admitted_at = 0});
  }
  caches_.reserve(n);
  tokens_.reserve(n);
  decoding_.reserve(n);
}

std::size_t BatchEngine::running() const {
  std::size_t n = 0;
  for (const auto& s : sequences_) {
    n += s->idle() ? 0 : 1;
  }
  return n;
}

Expected<std::optional<SlotId>> BatchEngine::admit(std::span<const std::int32_t> prompt,
                                                   const GenerateOptions& opts, TextSink on_text,
                                                   DoneSink done) {
  // The free slot that already holds the longest prefix of this prompt, so a client that
  // continues its conversation prefills only its new turn.
  std::optional<SlotId> best;
  std::size_t best_shared = 0;
  for (SlotId i = 0; i < sequences_.size(); ++i) {
    if (!sequences_[i]->idle()) {
      continue;
    }
    const std::size_t shared = sequences_[i]->shared_prefix(prompt);
    if (!best || shared > best_shared) {
      best = i;
      best_shared = shared;
    }
  }
  if (!best) {
    return std::optional<SlotId>{};
  }
  const auto started = sequences_[*best]->start(prompt, opts, std::move(on_text));
  if (!started) {
    return started.error();
  }
  slots_[*best].done = std::move(done);
  slots_[*best].admitted_at = ++admissions_;
  return best;
}

void BatchEngine::finish(SlotId id) {
  const Sequence& seq = *sequences_[id];
  if (slots_[id].done) {
    DoneSink done = std::move(slots_[id].done);
    slots_[id].done = {};
    done(seq.finish(), seq.stats());
  }
}

Expected<std::size_t> BatchEngine::step() {
  std::size_t finished = 0;
  // One prefill chunk for the oldest sequence that still has prompt left.
  std::optional<SlotId> oldest;
  for (SlotId i = 0; i < sequences_.size(); ++i) {
    if (sequences_[i]->state() == Sequence::State::Prefill &&
        (!oldest || slots_[i].admitted_at < slots_[*oldest].admitted_at)) {
      oldest = i;
    }
  }
  if (oldest) {
    const auto chunk = sequences_[*oldest]->prefill_chunk();
    if (!chunk) {
      slots_[*oldest].done = {};
      return chunk.error();
    }
  }
  // One decode step for everyone in decode: apply the end rules, batch the survivors.
  caches_.clear();
  tokens_.clear();
  decoding_.clear();
  for (SlotId i = 0; i < sequences_.size(); ++i) {
    Sequence& seq = *sequences_[i];
    if (seq.state() != Sequence::State::Decode) {
      continue;
    }
    if (!seq.accept_pending()) {
      finish(i);
      ++finished;
      continue;
    }
    caches_.push_back(&seq.cache());
    tokens_.push_back(seq.pending());
    decoding_.push_back(i);
  }
  if (decoding_.empty()) {
    return finished;
  }
  for (const SlotId i : decoding_) {
    sequences_[i]->fed();
  }
  const auto logits = model_.forward_batch(caches_, tokens_);
  if (!logits) {
    // The pass failed under every sequence in it: they end without a done call, and the
    // caller, which gets the error, tells their owners.
    for (const SlotId i : decoding_) {
      sequences_[i]->abandon();
      slots_[i].done = {};
    }
    return logits.error();
  }
  const std::size_t n_vocab = model_.config().n_vocab;
  for (std::size_t r = 0; r < decoding_.size(); ++r) {
    sequences_[decoding_[r]]->observe(logits->subspan(r * n_vocab, n_vocab));
  }
  ++decode_steps_;
  if (decoding_.size() > 1) {
    ++batched_steps_;
  }
  return finished;
}

}  // namespace axon::engine
