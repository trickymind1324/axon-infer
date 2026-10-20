#include "engine/engine.h"

#include <utility>

namespace axon::engine {

double Stats::prefill_tok_s() const {
  const std::size_t fed = prompt_tokens - reused_tokens;
  return prefill_s > 0.0 && fed > 0 ? static_cast<double>(fed) / prefill_s : 0.0;
}

double Stats::decode_tok_s() const {
  return decode_s > 0.0 && generated_tokens > 0 ? static_cast<double>(generated_tokens) / decode_s
                                                : 0.0;
}

Engine::Engine(model::LlamaModel& model) : model_(model), sequence_(model) {}

void Engine::clear() {
  sequence_.clear();
}

// The single-sequence loop over the shared Sequence: prefill in chunks, then feed each
// accepted token through forward_one and sample the next, until a rule ends it.
Expected<std::vector<std::int32_t>>
Engine::generate(std::span<const std::int32_t> prompt, const GenerateOptions& opts,
                 const std::function<void(std::string_view)>& on_text) {
  const auto started = sequence_.start(prompt, opts, on_text);
  if (!started) {
    return started.error();
  }
  while (sequence_.state() == Sequence::State::Prefill) {
    const auto chunk = sequence_.prefill_chunk();
    if (!chunk) {
      return chunk.error();
    }
  }
  const std::size_t n_vocab = model_.config().n_vocab;
  while (sequence_.accept_pending()) {
    const std::int32_t token = sequence_.pending();
    sequence_.fed();
    auto out = model_.forward_one(sequence_.cache(), token);
    if (!out) {
      return out.error();
    }
    sequence_.observe(out->last(n_vocab));
  }
  return std::vector<std::int32_t>(sequence_.generated().begin(), sequence_.generated().end());
}

}  // namespace axon::engine
