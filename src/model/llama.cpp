#include "model/llama.h"

#include "core/axt.h"
#include "core/check.h"

#include "kernels/kernels.h"
#include "model/activations.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace axon::model {

namespace {

// y[n_out] = W[n_out][n_in] x[n_in] for one activation row, over output rows [begin, end) of
// the weight. Quantized weights take the row's Q8_0 blocks in `xq` unless `exact` asks for the
// float path. Each output element is computed by exactly one call with the same instruction
// sequence whatever the split, which is what keeps threaded results bit-identical.
void matmul_columns(const Tensor& w, const float* x, const BlockQ8_0* xq, float* y,
                    std::size_t begin, std::size_t end, bool exact) {
  const auto& k = kernels::kernels();
  const auto n_in = static_cast<std::size_t>(w.ne0);
  const std::size_t n_out = end - begin;
  const auto* base = static_cast<const std::byte*>(w.data) + begin * w.row_bytes;
  float* out = y + begin;
  switch (w.type) {
  case DType::F32:
    k.matmul_f32(reinterpret_cast<const float*>(base), x, out, n_out, n_in);
    return;
  case DType::F16:
    k.matmul_f16(reinterpret_cast<const std::uint16_t*>(base), x, out, n_out, n_in);
    return;
  case DType::BF16:
    k.matmul_bf16(reinterpret_cast<const std::uint16_t*>(base), x, out, n_out, n_in);
    return;
  case DType::Q8_0:
    if (exact) {
      k.matmul_q8_0(reinterpret_cast<const BlockQ8_0*>(base), x, out, n_out, n_in);
    } else {
      k.matmul_q8_0_q8(reinterpret_cast<const BlockQ8_0*>(base), xq, out, n_out, n_in);
    }
    return;
  case DType::Q4_0:
    if (exact) {
      k.matmul_q4_0(reinterpret_cast<const BlockQ4_0*>(base), x, out, n_out, n_in);
    } else {
      k.matmul_q4_0_q8(reinterpret_cast<const BlockQ4_0*>(base), xq, out, n_out, n_in);
    }
    return;
  case DType::Q4_1:
    if (exact) {
      k.matmul_q4_1(reinterpret_cast<const BlockQ4_1*>(base), x, out, n_out, n_in);
    } else {
      k.matmul_q4_1_q8(reinterpret_cast<const BlockQ4_1*>(base), xq, out, n_out, n_in);
    }
    return;
  case DType::Q6_K:
    // Float activations only: the super-block format has no integer path yet.
    k.matmul_q6_k(reinterpret_cast<const BlockQ6_K*>(base), x, out, n_out, n_in);
    return;
  default:
    AXON_CHECK(false, "matmul: unbound dtype {}", dtype_name(w.type));
  }
}

bool needs_q8_activations(const Tensor& w, bool exact) {
  return !exact && (w.type == DType::Q8_0 || w.type == DType::Q4_0 || w.type == DType::Q4_1);
}

void add_bias(float* rows, const Tensor& bias, std::size_t n_rows) {
  const auto* b = static_cast<const float*>(bias.data);
  const auto n = static_cast<std::size_t>(bias.ne0);
  for (std::size_t r = 0; r < n_rows; ++r) {
    float* row = rows + r * n;
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

// Writes one activation as <dump_dir>/<name>.axt with shape [rows][cols]. A no-op unless a
// dump was requested; the arena path never touches the heap.
Expected<bool> dump(const ForwardOptions& opts, bool wanted, const char* name,
                    std::span<const float> data, std::size_t rows) {
  if (!wanted) {
    return true;
  }
  const std::uint64_t dims[] = {rows, data.size() / rows};
  auto written = write_axt_f32(opts.dump_dir / fmt::format("{}.axt", name), dims, data);
  if (!written) {
    return written.error();
  }
  return true;
}

std::size_t sum_weight_bytes(const LlamaWeights& w, const LlamaConfig& cfg) {
  std::size_t total = w.token_embd.row_bytes;  // one embedding row per token
  total += w.output_norm.total_bytes + w.output.total_bytes;
  if (w.rope_freqs) {
    total += w.rope_freqs->total_bytes;
  }
  for (const auto& l : w.layers) {
    total += l.attn_norm.total_bytes + l.wq.total_bytes + l.wk.total_bytes + l.wv.total_bytes +
             l.wo.total_bytes + l.ffn_norm.total_bytes + l.w_gate.total_bytes + l.w_up.total_bytes +
             l.w_down.total_bytes;
    if (l.bq && l.bk && l.bv) {
      total += l.bq->total_bytes + l.bk->total_bytes + l.bv->total_bytes;
    }
  }
  static_cast<void>(cfg);
  return total;
}

}  // namespace

LlamaModel::LlamaModel(GGUFFile file, LlamaConfig cfg, std::unique_ptr<Tokenizer> tokenizer,
                       LlamaWeights weights, LoadOptions options)
    : file_(std::move(file)), cfg_(std::move(cfg)), tokenizer_(std::move(tokenizer)),
      weights_(std::move(weights)), max_seq_(options.max_seq), max_ctx_(options.max_ctx),
      exact_activations_(options.exact_activations), pool_(options.n_threads),
      weight_bytes_(sum_weight_bytes(weights_, cfg_)),
      arena_(cfg_.activation_bytes(max_seq_, max_ctx_, static_cast<std::size_t>(pool_.size()))) {}

Expected<std::unique_ptr<LlamaModel>> LlamaModel::load(const std::filesystem::path& path,
                                                       LoadOptions options) {
  if (options.max_seq == 0) {
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
  if (options.max_ctx < options.max_seq) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("max_ctx {} is smaller than max_seq {}", options.max_ctx, options.max_seq));
  }
  // A cache longer than the training context is never useful, and a per-call chunk longer
  // than the cache can never be fed, so both are clamped rather than refused.
  if (options.max_ctx > cfg->n_ctx_train) {
    options.max_ctx = cfg->n_ctx_train;
  }
  if (options.max_seq > options.max_ctx) {
    options.max_seq = options.max_ctx;
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
      std::move(*file), std::move(*cfg), std::move(*tokenizer), std::move(*weights), options));
}

Expected<std::span<const float>> LlamaModel::forward(std::span<const std::int32_t> tokens,
                                                     const ForwardOptions& opts) {
  return run(tokens, nullptr, opts);
}

Expected<std::span<const float>> LlamaModel::forward_prefill(KVCache& cache,
                                                             std::span<const std::int32_t> tokens) {
  return run(tokens, &cache, ForwardOptions{});
}

Expected<std::span<const float>> LlamaModel::forward_one(KVCache& cache, std::int32_t token) {
  return run(std::span<const std::int32_t>(&token, 1), &cache, ForwardOptions{});
}

// The one forward pass. Without a cache, keys and values live in the arena and the attention
// spans the new tokens only. With a cache, new keys and values are written straight into the
// cache rows after the ones already there, and the attention spans the whole cache.
//
// Threads split output columns of every matmul, heads of the attention and rows of the
// activation quantization. No partial sum ever crosses a thread, so the logits do not depend
// on the thread count. Work is claimed in chunks rather than split statically, because the
// M1's efficiency cores are a few times slower than its performance cores and a static split
// makes every launch wait for the slowest share.
Expected<std::span<const float>> LlamaModel::run(std::span<const std::int32_t> tokens,
                                                 KVCache* cache, const ForwardOptions& opts) {
  const std::size_t seq = tokens.size();
  if (seq == 0) {
    return make_error(ErrorCode::InvalidArgument, "forward: empty token sequence");
  }
  if (seq > max_seq_) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("forward: {} tokens exceed max_seq {}", seq, max_seq_));
  }
  for (const std::int32_t id : tokens) {
    if (id < 0 || std::cmp_greater_equal(id, cfg_.n_vocab)) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("forward: token id {} outside [0, {})", id, cfg_.n_vocab));
    }
  }
  const std::size_t n_past = cache != nullptr ? cache->pos() : 0;
  if (cache != nullptr) {
    if (!cache->matches(cfg_)) {
      return make_error(ErrorCode::InvalidArgument,
                        "forward: the cache was built for a different model");
    }
    if (n_past + seq > cache->max_ctx()) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("context overflow: {} cached + {} new tokens exceed {}", n_past,
                                    seq, cache->max_ctx()));
    }
  }

  const auto& k = kernels::kernels();
  const std::size_t n_embd = cfg_.n_embd;
  const std::size_t n_ff = cfg_.n_ff;
  const std::size_t head_dim = cfg_.head_dim;
  const std::size_t n_head = cfg_.n_head;
  const std::size_t n_head_kv = cfg_.n_head_kv;
  const std::size_t kv_row = cfg_.n_embd_kv();
  const std::size_t total = n_past + seq;
  const bool dumping = opts.dump_layer != ForwardOptions::kNoDump;
  const auto n_threads = static_cast<std::size_t>(pool_.size());
  const kernels::RopeParams rope{
      .head_dim = head_dim,
      .rotary_dim = cfg_.rotary_dim,
      .theta = cfg_.rope_theta,
      .style = cfg_.rope_style,
      .freq_factors =
          weights_.rope_freqs ? static_cast<const float*>(weights_.rope_freqs->data) : nullptr};
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  arena_.reset();
  Activations a = allocate_activations(arena_, cfg_, seq, total, cache == nullptr, n_threads);

  // Y[seq][n_out] = X[seq][n_in] W^T with output columns split across threads.
  const auto mm = [&](const Tensor& w, const float* x, float* y) {
    const auto n_in = static_cast<std::size_t>(w.ne0);
    const auto n_out = static_cast<std::size_t>(w.ne1);
    const std::size_t nb = n_in / kQK8_0;
    const bool quantize = needs_q8_activations(w, exact_activations_);
    if (quantize) {
      pool_.parallel_for(0, seq, 4, [&](std::size_t begin, std::size_t end, int) {
        for (std::size_t r = begin; r < end; ++r) {
          k.quantize_row_q8_0(x + r * n_in, a.xq.data() + r * nb, n_in);
        }
      });
    }
    // About eight chunks per thread: enough to balance fast and slow cores, few enough that
    // the atomic claim per chunk is noise next to the columns it covers.
    const std::size_t chunk = std::max<std::size_t>(16, n_out / (n_threads * 8));
    pool_.parallel_for(0, n_out, chunk, [&](std::size_t begin, std::size_t end, int) {
      for (std::size_t r = 0; r < seq; ++r) {
        matmul_columns(w, x + r * n_in, quantize ? a.xq.data() + r * nb : nullptr, y + r * n_out,
                       begin, end, exact_activations_);
      }
    });
  };

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
    // Where this call's keys and values go, and where the attention reads all of them from.
    float* k_base = cache != nullptr ? cache->k(l) : a.k.data();
    float* v_base = cache != nullptr ? cache->v(l) : a.v.data();
    float* k_new = k_base + n_past * kv_row;
    float* v_new = v_base + n_past * kv_row;
    AXON_DUMP(dump_this, "x_in", a.x);

    rmsnorm_rows(a.x, w.attn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "attn_norm_out", a.h);
    mm(w.wq, a.h.data(), a.q.data());
    mm(w.wk, a.h.data(), k_new);
    mm(w.wv, a.h.data(), v_new);
    if (w.bq && w.bk && w.bv) {
      add_bias(a.q.data(), *w.bq, seq);
      add_bias(k_new, *w.bk, seq);
      add_bias(v_new, *w.bv, seq);
    }
    for (std::size_t i = 0; i < seq; ++i) {
      k.rope(a.q.data() + i * n_head * head_dim, n_head, n_past + i, rope);
      k.rope(k_new + i * kv_row, n_head_kv, n_past + i, rope);
    }
    AXON_DUMP(dump_this, "q", a.q);
    AXON_DUMP(dump_this, "k", std::span<const float>(k_new, seq * kv_row));
    AXON_DUMP(dump_this, "v", std::span<const float>(v_new, seq * kv_row));
    pool_.parallel_for(0, n_head, 1, [&](std::size_t begin, std::size_t end, int tid) {
      float* scores = a.scores.data() + static_cast<std::size_t>(tid) * seq * total;
      k.attention_kv_heads(a.q.data(), k_base, v_base, a.attn.data(), scores, seq, n_past, n_head,
                           n_head_kv, head_dim, scale, begin, end);
    });
    AXON_DUMP(dump_this, "attn_out", a.attn);
    mm(w.wo, a.attn.data(), a.h.data());
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_mid", a.x);

    rmsnorm_rows(a.x, w.ffn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "ffn_norm_out", a.h);
    mm(w.w_gate, a.h.data(), a.gate.data());
    mm(w.w_up, a.h.data(), a.up.data());
    k.silu_mul(a.gate.data(), a.up.data(), a.gate.data(), seq * n_ff);
    mm(w.w_down, a.gate.data(), a.h.data());
    AXON_DUMP(dump_this, "ffn_out", a.h);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_out", a.x);
  }

  rmsnorm_rows(a.x, weights_.output_norm, a.h, seq, cfg_.rms_eps);
  mm(weights_.output, a.h.data(), a.logits.data());
  AXON_DUMP(dumping, "logits", a.logits);
#undef AXON_DUMP
  if (cache != nullptr) {
    cache->advance(seq);
  }
  return std::span<const float>(a.logits);
}

}  // namespace axon::model
