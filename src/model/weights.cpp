#include "model/weights.h"

#include <fmt/format.h>

#include <cstdint>
#include <string>

namespace axon::model {

namespace {

enum class Kind : std::uint8_t { Norm, Matmul, Bias };

// Looks up one tensor and checks it is exactly what the forward pass will index.
Expected<Tensor> bind_tensor(const GGUFFile& file, const std::string& name, Kind kind,
                             std::uint64_t ne0, std::uint64_t ne1) {
  const TensorInfo* info = file.tensor(name);
  if (info == nullptr) {
    return make_error(ErrorCode::InvalidArgument, fmt::format("{}: missing", name));
  }
  const std::uint32_t want_dims = kind == Kind::Matmul ? 2 : 1;
  if (info->n_dims != want_dims || info->ne(0) != ne0 || info->ne(1) != ne1) {
    std::string expected = want_dims == 2 ? fmt::format("{}x{}", ne0, ne1) : fmt::format("{}", ne0);
    std::string found = fmt::format("{}", info->ne(0));
    for (std::uint32_t d = 1; d < info->n_dims; ++d) {
      found += fmt::format("x{}", info->ne(d));
    }
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("{}: expected shape {}, found {}", name, expected, found));
  }
  switch (kind) {
  case Kind::Norm:
  case Kind::Bias:
    if (info->dtype != DType::F32) {
      return make_error(ErrorCode::Unsupported, fmt::format("{}: expected dtype F32, found {}",
                                                            name, dtype_name(info->dtype)));
    }
    break;
  case Kind::Matmul: {
    const bool block32 =
        info->dtype == DType::Q8_0 || info->dtype == DType::Q4_0 || info->dtype == DType::Q4_1;
    const bool block256 = info->dtype == DType::Q6_K;
    if (info->dtype != DType::F32 && info->dtype != DType::F16 && info->dtype != DType::BF16 &&
        !block32 && !block256) {
      return make_error(ErrorCode::Unsupported,
                        fmt::format("{}: dtype {} has no kernels", name, dtype_name(info->dtype)));
    }
    const std::uint64_t block = block32 ? kQK8_0 : block256 ? kQK_K : 1;
    if (ne0 % block != 0) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("{}: {} rows need a multiple of {} elements, found {}", name,
                                    dtype_name(info->dtype), block, ne0));
    }
    break;
  }
  }
  return Tensor{.data = info->data, .type = info->dtype, .ne0 = ne0, .ne1 = ne1};
}

}  // namespace

Expected<LlamaWeights> LlamaWeights::bind(const GGUFFile& file, const LlamaConfig& cfg) {
  LlamaWeights w;
  const std::uint64_t d = cfg.n_embd;
  const std::uint64_t dq = cfg.n_embd_q();
  const std::uint64_t dkv = cfg.n_embd_kv();
  const std::uint64_t ff = cfg.n_ff;

#define AXON_BIND(target, name, kind, ne0, ne1)                                                    \
  do {                                                                                             \
    auto bound = bind_tensor(file, name, kind, ne0, ne1);                                          \
    if (!bound) {                                                                                  \
      return bound.error();                                                                        \
    }                                                                                              \
    (target) = *bound;                                                                             \
  } while (false)

  AXON_BIND(w.token_embd, "token_embd.weight", Kind::Matmul, d, cfg.n_vocab);
  AXON_BIND(w.output_norm, "output_norm.weight", Kind::Norm, d, 1);
  if (cfg.tie_embeddings) {
    w.output = w.token_embd;
  } else {
    AXON_BIND(w.output, "output.weight", Kind::Matmul, d, cfg.n_vocab);
  }
  if (cfg.rope_scaling == RopeScaling::Llama3) {
    Tensor factors;
    AXON_BIND(factors, "rope_freqs.weight", Kind::Norm, cfg.rotary_dim / 2, 1);
    w.rope_freqs = factors;
  }

  w.layers.resize(cfg.n_layer);
  for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
    LayerWeights& layer = w.layers[i];
    const std::string p = fmt::format("blk.{}.", i);
    AXON_BIND(layer.attn_norm, p + "attn_norm.weight", Kind::Norm, d, 1);
    AXON_BIND(layer.wq, p + "attn_q.weight", Kind::Matmul, d, dq);
    AXON_BIND(layer.wk, p + "attn_k.weight", Kind::Matmul, d, dkv);
    AXON_BIND(layer.wv, p + "attn_v.weight", Kind::Matmul, d, dkv);
    AXON_BIND(layer.wo, p + "attn_output.weight", Kind::Matmul, dq, d);
    AXON_BIND(layer.ffn_norm, p + "ffn_norm.weight", Kind::Norm, d, 1);
    AXON_BIND(layer.w_gate, p + "ffn_gate.weight", Kind::Matmul, d, ff);
    AXON_BIND(layer.w_up, p + "ffn_up.weight", Kind::Matmul, d, ff);
    AXON_BIND(layer.w_down, p + "ffn_down.weight", Kind::Matmul, ff, d);
    if (cfg.attn_bias) {
      Tensor bq;
      Tensor bk;
      Tensor bv;
      AXON_BIND(bq, p + "attn_q.bias", Kind::Bias, dq, 1);
      AXON_BIND(bk, p + "attn_k.bias", Kind::Bias, dkv, 1);
      AXON_BIND(bv, p + "attn_v.bias", Kind::Bias, dkv, 1);
      layer.bq = bq;
      layer.bk = bk;
      layer.bv = bv;
    }
  }
#undef AXON_BIND
  return w;
}

}  // namespace axon::model
