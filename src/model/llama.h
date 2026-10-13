#pragma once

#include "axon/tokenizer.h"
#include "core/arena.h"
#include "core/expected.h"
#include "core/gguf.h"
#include "core/threadpool.h"

#include "kernels/kernels.h"
#include "model/config.h"
#include "model/kv_cache.h"
#include "model/weights.h"

#include <array>
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
  // Threads for the forward pass. 0 means one per physical core. Any count gives logits
  // bit-identical to one thread, because work is split so no reduction crosses threads.
  int n_threads = 0;
  // Which kernel table runs the forward pass. Auto picks the best the host supports.
  kernels::KernelSet kernels = kernels::KernelSet::Auto;
  // Microseconds an idle worker thread spins before sleeping, see ThreadPoolOptions.
  int spin_budget_us = ThreadPoolOptions::kDefaultSpinBudgetUs;
  // Key/value cache storage. F32 keeps cached decoding bit-identical to the cache-free pass.
  // F16 halves the cache and rounds each key and value once when it is written.
  CacheType cache = CacheType::F32;
};

// Where a forward call spent its wall-clock time, by launch kind. Filled only when a
// ForwardOptions names a profile, which costs one clock read per launch and nothing otherwise.
// Seconds accumulate across calls, so a caller can profile a whole prompt in chunks.
struct ForwardProfile {
  enum Slot : std::uint8_t {
    kQuantize,    // activation rows quantized to Q8_0 blocks
    kQkv,         // the fused q, k and v projections
    kAttention,   // attention over the cache or the staged keys and values
    kOutputProj,  // the attention output projection
    kGateUp,      // the fused gate and up projections with the SiLU product
    kDown,        // the down projection
    kLogits,      // the output matrix
    kOther,       // embedding, norms, RoPE, biases, residual adds, fp16 cache rounding
    kSlots
  };
  std::array<double, kSlots> seconds{};
  [[nodiscard]] double total() const noexcept {
    double sum = 0.0;
    for (const double s : seconds) {
      sum += s;
    }
    return sum;
  }
  static const char* name(Slot slot) noexcept;
};

struct ForwardOptions {
  static constexpr std::size_t kNoDump = std::numeric_limits<std::size_t>::max();
  // Write this layer's activations (plus the embeddings and logits) as .axt files.
  std::size_t dump_layer = kNoDump;
  std::filesystem::path dump_dir;
  // Accumulate the time per launch kind here. Null leaves the hot path untouched.
  ForwardProfile* profile = nullptr;
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
                                                   std::span<const std::int32_t> tokens,
                                                   const ForwardOptions& opts = {});
  // One token at position cache.pos(). Logits are [1][n_vocab].
  Expected<std::span<const float>> forward_one(KVCache& cache, std::int32_t token);

  // A cache sized for this model's max_ctx.
  [[nodiscard]] KVCache new_cache() const { return KVCache(cfg_, max_ctx_, cache_type_); }
  [[nodiscard]] CacheType cache_type() const noexcept { return cache_type_; }

  [[nodiscard]] const LlamaConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const LlamaWeights& weights() const noexcept { return weights_; }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
  [[nodiscard]] const GGUFFile& file() const noexcept { return file_; }
  [[nodiscard]] std::size_t max_seq() const noexcept { return max_seq_; }
  [[nodiscard]] std::size_t max_ctx() const noexcept { return max_ctx_; }
  [[nodiscard]] bool exact_activations() const noexcept { return exact_activations_; }
  [[nodiscard]] int n_threads() const noexcept { return pool_.size(); }
  [[nodiscard]] const ThreadPool& pool() const noexcept { return pool_; }
  [[nodiscard]] const kernels::KernelTable& table() const noexcept { return *kernels_; }
  // Bytes of weights one forward pass reads per position, the number to hold against memory
  // bandwidth.
  [[nodiscard]] std::size_t weight_bytes_per_token() const noexcept { return weight_bytes_; }
  [[nodiscard]] std::size_t arena_high_water() const noexcept { return arena_.high_water(); }
  [[nodiscard]] std::size_t arena_capacity() const noexcept { return arena_.capacity(); }

private:
  LlamaModel(GGUFFile file, LlamaConfig cfg, std::unique_ptr<Tokenizer> tokenizer,
             LlamaWeights weights, LoadOptions options, const kernels::KernelTable* table);
  Expected<std::span<const float>> run(std::span<const std::int32_t> tokens, KVCache* cache,
                                       const ForwardOptions& opts);

  GGUFFile file_;  // owns the mapping every weight points into; declared first, destroyed last
  LlamaConfig cfg_;
  std::unique_ptr<Tokenizer> tokenizer_;
  LlamaWeights weights_;
  std::size_t max_seq_;
  std::size_t max_ctx_;
  CacheType cache_type_;
  bool exact_activations_;
  const kernels::KernelTable* kernels_;
  ThreadPool pool_;
  std::size_t weight_bytes_ = 0;
  Arena arena_;
};

}  // namespace axon::model
