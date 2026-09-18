#include "model/llama.h"

#include "core/axt.h"
#include "core/check.h"

#include "kernels/kernels.h"
#include "model/activations.h"

#include <fmt/format.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace axon::model {

namespace {

// Y[n_rows][ne1] = X[n_rows][ne0] W^T, picking the kernel by weight dtype.
void matmul_rows(const Tensor& w, const float* x, float* y, std::size_t n_rows) {
  const auto& k = kernels::kernels();
  const auto n_out = static_cast<std::size_t>(w.ne1);
  const auto n_in = static_cast<std::size_t>(w.ne0);
  switch (w.type) {
  case DType::F32:
    k.matmul_f32_rows(static_cast<const float*>(w.data), x, y, n_rows, n_out, n_in);
    return;
  case DType::F16:
    k.matmul_f16_rows(static_cast<const std::uint16_t*>(w.data), x, y, n_rows, n_out, n_in);
    return;
  case DType::BF16:
    k.matmul_bf16_rows(static_cast<const std::uint16_t*>(w.data), x, y, n_rows, n_out, n_in);
    return;
  default:
    AXON_CHECK(false, "matmul_rows: unbound dtype {}", dtype_name(w.type));
  }
}

void add_bias(std::span<float> rows, const Tensor& bias, std::size_t n_rows) {
  const auto* b = static_cast<const float*>(bias.data);
  const auto n = static_cast<std::size_t>(bias.ne0);
  for (std::size_t r = 0; r < n_rows; ++r) {
    float* row = rows.data() + r * n;
    for (std::size_t i = 0; i < n; ++i) {
      row[i] += b[i];
    }
  }
}

void add_rows(std::span<float> x, std::span<const float> y) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] += y[i];
  }
}

void rmsnorm_rows(std::span<const float> x, const Tensor& weight, std::span<float> out,
                  std::size_t n_rows, float eps) {
  const auto& k = kernels::kernels();
  const auto n = static_cast<std::size_t>(weight.ne0);
  for (std::size_t r = 0; r < n_rows; ++r) {
    k.rmsnorm(x.data() + r * n, static_cast<const float*>(weight.data), out.data() + r * n, n, eps);
  }
}

// Writes one activation as <dump_dir>/<name>.axt with shape [seq][cols]. A no-op unless a
// dump was requested; the arena path never touches the heap.
Expected<bool> dump(const ForwardOptions& opts, bool wanted, const char* name,
                    std::span<const float> data, std::size_t seq) {
  if (!wanted) {
    return true;
  }
  const std::uint64_t dims[] = {seq, data.size() / seq};
  auto written = write_axt_f32(opts.dump_dir / fmt::format("{}.axt", name), dims, data);
  if (!written) {
    return written.error();
  }
  return true;
}

}  // namespace

LlamaModel::LlamaModel(GGUFFile file, LlamaConfig cfg, std::unique_ptr<Tokenizer> tokenizer,
                       LlamaWeights weights, std::size_t max_seq)
    : file_(std::move(file)), cfg_(std::move(cfg)), tokenizer_(std::move(tokenizer)),
      weights_(std::move(weights)), max_seq_(max_seq), arena_(cfg_.activation_bytes(max_seq)) {}

Expected<std::unique_ptr<LlamaModel>> LlamaModel::load(const std::filesystem::path& path,
                                                       std::size_t max_seq) {
  if (max_seq == 0) {
    return make_error(ErrorCode::InvalidArgument, "max_seq: expected at least 1, found 0");
  }
  auto file = GGUFFile::open(path);
  if (!file) {
    return file.error();
  }
  auto cfg = LlamaConfig::from_gguf(*file);
  if (!cfg) {
    return make_error(cfg.error().code, fmt::format("{}: {}", path.string(), cfg.error().message));
  }
  auto tokenizer = Tokenizer::from_gguf(*file);
  if (!tokenizer) {
    return make_error(tokenizer.error().code,
                      fmt::format("{}: {}", path.string(), tokenizer.error().message));
  }
  auto weights = LlamaWeights::bind(*file, *cfg);
  if (!weights) {
    return make_error(weights.error().code,
                      fmt::format("{}: {}", path.string(), weights.error().message));
  }
  return std::unique_ptr<LlamaModel>(new LlamaModel(
      std::move(*file), std::move(*cfg), std::move(*tokenizer), std::move(*weights), max_seq));
}

Expected<std::span<const float>> LlamaModel::forward(std::span<const std::int32_t> tokens,
                                                     const ForwardOptions& opts) {
  const std::size_t seq = tokens.size();
  if (seq == 0) {
    return make_error(ErrorCode::InvalidArgument, "forward: empty token sequence");
  }
  if (seq > max_seq_) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("forward: {} tokens exceed max_seq {}", seq, max_seq_));
  }
  for (const std::int32_t id : tokens) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= cfg_.n_vocab) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("forward: token id {} outside [0, {})", id, cfg_.n_vocab));
    }
  }

  const auto& k = kernels::kernels();
  const std::size_t n_embd = cfg_.n_embd;
  const std::size_t n_ff = cfg_.n_ff;
  const std::size_t head_dim = cfg_.head_dim;
  const std::size_t n_head = cfg_.n_head;
  const std::size_t n_head_kv = cfg_.n_head_kv;
  const bool dumping = opts.dump_layer != ForwardOptions::kNoDump;
  const kernels::RopeParams rope{
      .head_dim = head_dim,
      .rotary_dim = cfg_.rotary_dim,
      .theta = cfg_.rope_theta,
      .style = cfg_.rope_style,
      .freq_factors =
          weights_.rope_freqs ? static_cast<const float*>(weights_.rope_freqs->data) : nullptr};
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  arena_.reset();
  Activations a = allocate_activations(arena_, cfg_, seq);

#define AXON_DUMP(wanted, name, span)                                                              \
  do {                                                                                             \
    auto dumped = dump(opts, (wanted), (name), (span), seq);                                       \
    if (!dumped) {                                                                                 \
      return dumped.error();                                                                       \
    }                                                                                              \
  } while (false)

  for (std::size_t i = 0; i < seq; ++i) {
    k.embedding_row(weights_.token_embd.type, weights_.token_embd.data, n_embd, tokens[i],
                    a.x.data() + i * n_embd);
  }
  AXON_DUMP(dumping, "embed", a.x);

  for (std::size_t l = 0; l < cfg_.n_layer; ++l) {
    const LayerWeights& w = weights_.layers[l];
    const bool dump_this = dumping && opts.dump_layer == l;
    AXON_DUMP(dump_this, "x_in", a.x);

    rmsnorm_rows(a.x, w.attn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "attn_norm_out", a.h);
    matmul_rows(w.wq, a.h.data(), a.q.data(), seq);
    matmul_rows(w.wk, a.h.data(), a.k.data(), seq);
    matmul_rows(w.wv, a.h.data(), a.v.data(), seq);
    if (w.bq) {
      add_bias(a.q, *w.bq, seq);
      add_bias(a.k, *w.bk, seq);
      add_bias(a.v, *w.bv, seq);
    }
    for (std::size_t i = 0; i < seq; ++i) {
      k.rope(a.q.data() + i * n_head * head_dim, n_head, i, rope);
      k.rope(a.k.data() + i * n_head_kv * head_dim, n_head_kv, i, rope);
    }
    AXON_DUMP(dump_this, "q", a.q);
    AXON_DUMP(dump_this, "k", a.k);
    AXON_DUMP(dump_this, "v", a.v);
    k.attention(a.q.data(), a.k.data(), a.v.data(), a.attn.data(), a.scores.data(), seq, n_head,
                n_head_kv, head_dim, scale);
    AXON_DUMP(dump_this, "attn_out", a.attn);
    matmul_rows(w.wo, a.attn.data(), a.h.data(), seq);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_mid", a.x);

    rmsnorm_rows(a.x, w.ffn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "ffn_norm_out", a.h);
    matmul_rows(w.w_gate, a.h.data(), a.gate.data(), seq);
    matmul_rows(w.w_up, a.h.data(), a.up.data(), seq);
    k.silu_mul(a.gate.data(), a.up.data(), a.gate.data(), seq * n_ff);
    matmul_rows(w.w_down, a.gate.data(), a.h.data(), seq);
    AXON_DUMP(dump_this, "ffn_out", a.h);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_out", a.x);
  }

  rmsnorm_rows(a.x, weights_.output_norm, a.h, seq, cfg_.rms_eps);
  matmul_rows(weights_.output, a.h.data(), a.logits.data(), seq);
  AXON_DUMP(dumping, "logits", a.logits);
#undef AXON_DUMP
  return std::span<const float>(a.logits);
}

}  // namespace axon::model
