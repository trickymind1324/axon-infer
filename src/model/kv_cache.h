#pragma once

#include "core/arena.h"

#include "model/config.h"

#include <cstddef>

namespace axon::model {

enum class CacheType { F32 };  // F16 is reserved for a later memory pass

// Keys and values for every layer, preallocated for max_ctx positions and filled in order.
// Layout per layer is [max_ctx][n_head_kv][head_dim], the same shape the attention kernel
// reads, so appending a position is one contiguous row write and prefill over an empty cache
// is exactly the full-sequence attention. Storage is an anonymous mapping, so untouched
// context pages cost nothing until they are written.
class KVCache {
public:
  KVCache(const LlamaConfig& cfg, std::size_t max_ctx, CacheType type = CacheType::F32);

  // Layer l's rows. The first pos() rows are live.
  [[nodiscard]] float* k(std::size_t layer) noexcept { return layer_base(layer, 0); }
  [[nodiscard]] float* v(std::size_t layer) noexcept { return layer_base(layer, 1); }
  [[nodiscard]] const float* k(std::size_t layer) const noexcept { return layer_base(layer, 0); }
  [[nodiscard]] const float* v(std::size_t layer) const noexcept { return layer_base(layer, 1); }

  [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
  [[nodiscard]] std::size_t max_ctx() const noexcept { return max_ctx_; }
  [[nodiscard]] std::size_t n_layer() const noexcept { return n_layer_; }
  [[nodiscard]] std::size_t row_floats() const noexcept { return row_; }  // n_head_kv * head_dim
  [[nodiscard]] std::size_t bytes() const noexcept { return storage_.capacity(); }
  [[nodiscard]] std::size_t live_bytes() const noexcept {
    return n_layer_ * 2 * pos_ * row_ * sizeof(float);
  }
  [[nodiscard]] CacheType type() const noexcept { return type_; }
  // Whether this cache was built for a model with the same layer count and kv row width.
  [[nodiscard]] bool matches(const LlamaConfig& cfg) const noexcept;

  void clear() noexcept { pos_ = 0; }
  // Keeps the first `pos` positions. Fatal when pos exceeds pos().
  void truncate(std::size_t pos);
  // Marks n more positions as written. Fatal past max_ctx.
  void advance(std::size_t n);

private:
  float* layer_base(std::size_t layer, std::size_t which) const noexcept;

  std::size_t n_layer_;
  std::size_t max_ctx_;
  std::size_t row_;
  std::size_t pos_ = 0;
  CacheType type_;
  Arena storage_;
  float* base_;
};

}  // namespace axon::model
