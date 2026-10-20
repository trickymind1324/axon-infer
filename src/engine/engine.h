#pragma once

// The generation loop: feeds a prompt through the KV cache, reusing whatever prefix the cache
// already holds, then samples tokens one at a time and streams the decoded text.

#include "core/expected.h"

#include "engine/options.h"
#include "engine/sequence.h"
#include "model/llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace axon::engine {

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
  [[nodiscard]] const Stats& stats() const noexcept { return sequence_.stats(); }
  [[nodiscard]] std::span<const std::int32_t> history() const noexcept {
    return sequence_.history();
  }
  [[nodiscard]] model::LlamaModel& model() noexcept { return model_; }

private:
  model::LlamaModel& model_;
  Sequence sequence_;
};

}  // namespace axon::engine
