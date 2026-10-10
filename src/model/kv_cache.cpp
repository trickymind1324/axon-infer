#include "model/kv_cache.h"

#include "core/check.h"

namespace axon::model {

KVCache::KVCache(const LlamaConfig& cfg, std::size_t max_ctx, CacheType type)
    : n_layer_(cfg.n_layer), max_ctx_(max_ctx), row_(cfg.n_embd_kv()), type_(type),
      storage_(n_layer_ * 2 * max_ctx_ * row_ * (type == CacheType::F16 ? 2 : 4)),
      base_(static_cast<std::byte*>(
          storage_.alloc(n_layer_ * 2 * max_ctx_ * row_ * element_bytes()))) {
  AXON_CHECK(max_ctx > 0, "KVCache: max_ctx must be at least 1");
}

void* KVCache::layer_base(std::size_t layer, std::size_t which) const noexcept {
  AXON_DCHECK(layer < n_layer_, "KVCache: layer {} of {}", layer, n_layer_);
  return base_ + (layer * 2 + which) * max_ctx_ * row_ * element_bytes();
}

bool KVCache::matches(const LlamaConfig& cfg) const noexcept {
  return cfg.n_layer == n_layer_ && cfg.n_embd_kv() == row_;
}

void KVCache::truncate(std::size_t pos) {
  AXON_CHECK(pos <= pos_, "KVCache: cannot truncate to {} with {} positions cached", pos, pos_);
  pos_ = pos;
}

void KVCache::advance(std::size_t n) {
  AXON_CHECK(pos_ + n <= max_ctx_, "KVCache: {} + {} positions exceed max_ctx {}", pos_, n,
             max_ctx_);
  pos_ += n;
}

}  // namespace axon::model
