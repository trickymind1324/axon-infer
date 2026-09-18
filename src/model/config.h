#pragma once

#include "core/expected.h"
#include "core/gguf.h"

#include "kernels/kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace axon::model {

enum class RopeScaling { None, Llama3 };

// Everything the forward pass needs to know about a model, read from GGUF metadata and the
// tensor table. Two architectures share the LLaMA block: "llama" (Llama 2/3, SmolLM2,
// TinyLlama) and "qwen2" (NeoX-style RoPE and Q/K/V bias).
struct LlamaConfig {
  std::string arch;
  std::uint32_t n_layer = 0;
  std::uint32_t n_embd = 0;
  std::uint32_t n_head = 0;
  std::uint32_t n_head_kv = 0;
  std::uint32_t n_ff = 0;
  std::uint32_t n_vocab = 0;
  std::uint32_t n_ctx_train = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t rotary_dim = 0;
  float rope_theta = 10000.0F;
  float rms_eps = 1e-5F;
  kernels::RopeStyle rope_style = kernels::RopeStyle::Normal;
  RopeScaling rope_scaling = RopeScaling::None;
  bool tie_embeddings = true;
  bool attn_bias = false;

  static Expected<LlamaConfig> from_gguf(const GGUFFile& file);

  // One line for logs and the CLI banner.
  [[nodiscard]] std::string describe() const;
  // Bytes the forward pass allocates from its arena for a sequence of max_seq tokens.
  [[nodiscard]] std::size_t activation_bytes(std::size_t max_seq) const;

  [[nodiscard]] std::uint32_t n_embd_kv() const { return n_head_kv * head_dim; }
  [[nodiscard]] std::uint32_t n_embd_q() const { return n_head * head_dim; }
};

}  // namespace axon::model
