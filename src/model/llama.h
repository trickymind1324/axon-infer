#pragma once

#include "axon/tokenizer.h"
#include "core/arena.h"
#include "core/expected.h"
#include "core/gguf.h"

#include "model/config.h"
#include "model/kv_cache.h"
#include "model/weights.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>

namespace axon::model {

struct LoadOptions {
  std::size_t max_seq = 256;   // new tokens per forward call, clamped to max_ctx
  std::size_t max_ctx = 2048;  // cache length, clamped to the training context
  // Quantized weights normally see activations quantized to Q8_0 blocks and an integer dot
  // product. This keeps the activations in float and dequantizes the weights instead, for
  // comparison.
  bool exact_activations = false;
};

struct ForwardOptions {
  static constexpr std::size_t kNoDump = std::numeric_limits<std::size_t>::max();
  // Write this layer's activations (plus the embeddings and logits) as .axt files.
  std::size_t dump_layer = kNoDump;
  std::filesystem::path dump_dir;
};

// A LLaMA-family model: memory-mapped weights, one arena for activations, and two forward
// paths on the same kernels. The cache-free pass over a whole sequence is the reference. The
// cached pass appends new positions to a KVCache and must reproduce the reference bit for bit.
class LlamaModel {
public:
  // Opens the file, reads the config, builds the tokenizer, binds the weights and sizes the
  // arena for the load options.
  static Expected<std::unique_ptr<LlamaModel>> load(const std::filesystem::path& path,
                                                    LoadOptions options = {});

  // Cache-free reference: logits for every position, [tokens.size()][n_vocab], valid until the
  // next forward call. Fails when the sequence is empty or longer than max_seq, or an id is
  // out of range. The only path that honours dump options.
  Expected<std::span<const float>> forward(std::span<const std::int32_t> tokens,
                                           const ForwardOptions& opts = {});

  // Appends tokens at positions cache.pos() onwards and returns their logits,
  // [tokens.size()][n_vocab]. Fails like forward, and also when the cache belongs to another
  // model or the new tokens would not fit.
  Expected<std::span<const float>> forward_prefill(KVCache& cache,
                                                   std::span<const std::int32_t> tokens);
  // One token at position cache.pos(). Logits are [1][n_vocab].
  Expected<std::span<const float>> forward_one(KVCache& cache, std::int32_t token);

  // A cache sized for this model's max_ctx.
  [[nodiscard]] KVCache new_cache() const { return KVCache(cfg_, max_ctx_); }

  [[nodiscard]] const LlamaConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const LlamaWeights& weights() const noexcept { return weights_; }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
  [[nodiscard]] const GGUFFile& file() const noexcept { return file_; }
  [[nodiscard]] std::size_t max_seq() const noexcept { return max_seq_; }
  [[nodiscard]] std::size_t max_ctx() const noexcept { return max_ctx_; }
  [[nodiscard]] bool exact_activations() const noexcept { return exact_activations_; }
  [[nodiscard]] std::size_t arena_high_water() const noexcept { return arena_.high_water(); }
  [[nodiscard]] std::size_t arena_capacity() const noexcept { return arena_.capacity(); }

private:
  LlamaModel(GGUFFile file, LlamaConfig cfg, std::unique_ptr<Tokenizer> tokenizer,
             LlamaWeights weights, LoadOptions options);
  Expected<std::span<const float>> run(std::span<const std::int32_t> tokens, KVCache* cache,
                                       const ForwardOptions& opts);

  GGUFFile file_;  // owns the mapping every weight points into; declared first, destroyed last
  LlamaConfig cfg_;
  std::unique_ptr<Tokenizer> tokenizer_;
  LlamaWeights weights_;
  std::size_t max_seq_;
  std::size_t max_ctx_;
  bool exact_activations_;
  Arena arena_;
};

}  // namespace axon::model
