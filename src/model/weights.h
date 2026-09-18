#pragma once

#include "core/dtype.h"
#include "core/expected.h"
#include "core/gguf.h"

#include "model/config.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace axon::model {

// A bound view into the memory-mapped file. Nothing is copied.
struct Tensor {
  const void* data = nullptr;
  DType type = DType::F32;
  std::uint64_t ne0 = 0;  // innermost (n_in for a matmul weight)
  std::uint64_t ne1 = 0;  // n_out for a matmul weight, 1 for vectors
};

struct LayerWeights {
  Tensor attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down;
  std::optional<Tensor> bq, bk, bv;  // Qwen2
};

struct LlamaWeights {
  Tensor token_embd;
  Tensor output_norm;
  Tensor output;                     // aliases token_embd when the config ties them
  std::optional<Tensor> rope_freqs;  // Llama 3.x factors, F32 [rotary_dim / 2]
  std::vector<LayerWeights> layers;

  // Checks every tensor's presence, dtype and full shape against the config.
  static Expected<LlamaWeights> bind(const GGUFFile& file, const LlamaConfig& cfg);
};

}  // namespace axon::model
