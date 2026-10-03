#include "model/config.h"

#include "model/activations.h"

#include <fmt/format.h>

namespace axon::model {

namespace {

Expected<std::uint32_t> get_u32(const GGUFFile& file, const std::string& key) {
  auto value = file.get_uint(key);
  if (!value) {
    return value.error();
  }
  if (*value > 0xFFFFFFFFULL) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{}: expected a 32-bit value, found {}", key, *value));
  }
  return static_cast<std::uint32_t>(*value);
}

}  // namespace

Expected<LlamaConfig> LlamaConfig::from_gguf(const GGUFFile& file) {
  LlamaConfig cfg;
  cfg.arch = std::string(file.architecture());
  if (cfg.arch == "llama") {
    cfg.rope_style = kernels::RopeStyle::Normal;
  } else if (cfg.arch == "qwen2") {
    cfg.rope_style = kernels::RopeStyle::NeoX;
  } else {
    return make_error(ErrorCode::Unsupported,
                      fmt::format("general.architecture: expected llama or qwen2, found '{}'",
                                  cfg.arch.empty() ? "(missing)" : cfg.arch));
  }
  const auto key = [&](const char* suffix) { return cfg.arch + "." + suffix; };

  struct Required {
    const char* suffix;
    std::uint32_t* field;
  };
  const Required required[] = {
      {.suffix = "block_count", .field = &cfg.n_layer},
      {.suffix = "embedding_length", .field = &cfg.n_embd},
      {.suffix = "attention.head_count", .field = &cfg.n_head},
      {.suffix = "feed_forward_length", .field = &cfg.n_ff},
      {.suffix = "context_length", .field = &cfg.n_ctx_train},
  };
  for (const auto& r : required) {
    auto value = get_u32(file, key(r.suffix));
    if (!value) {
      return value.error();
    }
    *r.field = *value;
  }
  cfg.n_head_kv =
      static_cast<std::uint32_t>(file.get_uint_or(key("attention.head_count_kv"), cfg.n_head));
  if (cfg.n_layer == 0 || cfg.n_embd == 0 || cfg.n_head == 0 || cfg.n_ff == 0) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{}: layer, embedding, head and feed-forward counts must be "
                                  "non-zero",
                                  cfg.arch));
  }
  if (cfg.n_head_kv == 0 || cfg.n_head % cfg.n_head_kv != 0) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{}: expected head_count ({}) to be a multiple of "
                                  "head_count_kv, found {}",
                                  key("attention.head_count_kv"), cfg.n_head, cfg.n_head_kv));
  }
  cfg.head_dim = static_cast<std::uint32_t>(
      file.get_uint_or(key("attention.key_length"), cfg.n_embd / cfg.n_head));
  if (cfg.head_dim == 0 || cfg.head_dim % 2 != 0) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("{}: expected an even head size, found {}", cfg.arch, cfg.head_dim));
  }
  cfg.rotary_dim =
      static_cast<std::uint32_t>(file.get_uint_or(key("rope.dimension_count"), cfg.head_dim));
  if (cfg.rotary_dim == 0 || cfg.rotary_dim % 2 != 0 || cfg.rotary_dim > cfg.head_dim) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{}: expected an even value up to the head size {}, found {}",
                                  key("rope.dimension_count"), cfg.head_dim, cfg.rotary_dim));
  }
  cfg.rope_theta = static_cast<float>(file.get_float_or(key("rope.freq_base"), 10000.0));
  cfg.rms_eps =
      static_cast<float>(file.get_float_or(key("attention.layer_norm_rms_epsilon"), 1e-5));

  const TensorInfo* embd = file.tensor("token_embd.weight");
  if (embd == nullptr) {
    return make_error(ErrorCode::InvalidArgument, "token_embd.weight: missing");
  }
  if (embd->n_dims != 2 || embd->ne(0) != cfg.n_embd) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("token_embd.weight: expected shape {}xN, found {}x{}", cfg.n_embd,
                                  embd->ne(0), embd->ne(1)));
  }
  cfg.n_vocab = static_cast<std::uint32_t>(embd->ne(1));
  if (const auto* declared = file.metadata(key("vocab_size")); declared != nullptr) {
    const auto n = declared->as_uint().value_or(0);
    if (n != cfg.n_vocab) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("{}: expected {} (the embedding rows), found {}",
                                    key("vocab_size"), cfg.n_vocab, n));
    }
  }
  if (const auto* tokens = file.metadata("tokenizer.ggml.tokens"); tokens != nullptr) {
    if (tokens->array_size() != cfg.n_vocab) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("tokenizer.ggml.tokens: expected {} entries (the embedding "
                                    "rows), found {}",
                                    cfg.n_vocab, tokens->array_size()));
    }
  }

  cfg.tie_embeddings = file.tensor("output.weight") == nullptr;
  cfg.attn_bias = file.tensor("blk.0.attn_q.bias") != nullptr;
  cfg.rope_scaling =
      file.tensor("rope_freqs.weight") != nullptr ? RopeScaling::Llama3 : RopeScaling::None;
  return cfg;
}

std::string LlamaConfig::describe() const {
  return fmt::format("{} layers {} embd {} heads {} kv {} ff {} vocab {} ctx {} rope {} {}{}{}{}",
                     arch, n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_ctx_train,
                     rope_style == kernels::RopeStyle::NeoX ? "neox" : "normal", rope_theta,
                     rope_scaling == RopeScaling::Llama3 ? " llama3-scaling" : "",
                     tie_embeddings ? " tied" : "", attn_bias ? " qkv-bias" : "");
}

std::size_t LlamaConfig::activation_bytes(std::size_t max_seq, std::size_t max_ctx,
                                          std::size_t n_threads) const {
  SizeCounter uncached;
  allocate_activations(uncached, *this, max_seq, max_seq, true, n_threads);
  SizeCounter cached;
  allocate_activations(cached, *this, max_seq, max_ctx, false, n_threads);
  return uncached.total > cached.total ? uncached.total : cached.total;
}

}  // namespace axon::model
