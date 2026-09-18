#pragma once

#include "axon/tokenizer.h"
#include "core/arena.h"
#include "core/expected.h"
#include "core/gguf.h"

#include "model/config.h"
#include "model/weights.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>

namespace axon::model {

struct ForwardOptions {
  static constexpr std::size_t kNoDump = std::numeric_limits<std::size_t>::max();
  // Write this layer's activations (plus the embeddings and logits) as .axt files.
  std::size_t dump_layer = kNoDump;
  std::filesystem::path dump_dir;
};

// A LLaMA-family model: memory-mapped weights, one arena for activations, and a
// full-sequence forward pass on the scalar kernel table. Single-threaded and cache-free by
// design in this phase; every later optimisation is measured against it.
class LlamaModel {
public:
  // Opens the file, reads the config, builds the tokenizer, binds the weights and sizes the
  // arena for sequences of up to max_seq tokens.
  static Expected<std::unique_ptr<LlamaModel>> load(const std::filesystem::path& path,
                                                    std::size_t max_seq);

  // Logits for every position, [tokens.size()][n_vocab], valid until the next forward call.
  // Fails when the sequence is empty or longer than max_seq, or an id is out of range.
  Expected<std::span<const float>> forward(std::span<const std::int32_t> tokens,
                                           const ForwardOptions& opts = {});

  [[nodiscard]] const LlamaConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const LlamaWeights& weights() const noexcept { return weights_; }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
  [[nodiscard]] const GGUFFile& file() const noexcept { return file_; }
  [[nodiscard]] std::size_t max_seq() const noexcept { return max_seq_; }
  [[nodiscard]] std::size_t arena_high_water() const noexcept { return arena_.high_water(); }
  [[nodiscard]] std::size_t arena_capacity() const noexcept { return arena_.capacity(); }

private:
  LlamaModel(GGUFFile file, LlamaConfig cfg, std::unique_ptr<Tokenizer> tokenizer,
             LlamaWeights weights, std::size_t max_seq);

  GGUFFile file_;  // owns the mapping every weight points into; declared first, destroyed last
  LlamaConfig cfg_;
  std::unique_ptr<Tokenizer> tokenizer_;
  LlamaWeights weights_;
  std::size_t max_seq_;
  Arena arena_;
};

}  // namespace axon::model
