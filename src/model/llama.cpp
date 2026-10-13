#include "model/llama.h"

#include "core/axt.h"
#include "core/check.h"

#include "kernels/kernels.h"
#include "model/activations.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>

namespace axon::model {

namespace {

// y[n_out] = W[n_out][n_in] x[n_in] for one activation row, over output rows [begin, end) of
// the weight. Quantized weights take the row's Q8_0 blocks in `xq` unless `exact` asks for the
// float path. Each output element is computed by exactly one call with the same instruction
// sequence whatever the split, which is what keeps threaded results bit-identical.
void matmul_columns(const kernels::KernelTable& k, const Tensor& w, const float* x,
                    const BlockQ8_0* xq, float* y, std::size_t begin, std::size_t end, bool exact) {
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

void rmsnorm_rows(const kernels::KernelTable& k, std::span<const float> x, const Tensor& weight,
                  std::span<float> out, std::size_t n_rows, float eps) {
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
                       LlamaWeights weights, LoadOptions options, const kernels::KernelTable* table)
    : file_(std::move(file)), cfg_(std::move(cfg)), tokenizer_(std::move(tokenizer)),
      weights_(std::move(weights)), max_seq_(options.max_seq), max_ctx_(options.max_ctx),
      cache_type_(options.cache), exact_activations_(options.exact_activations), kernels_(table),
      pool_(ThreadPoolOptions{.n_threads = options.n_threads,
                              .spin_budget = std::chrono::microseconds(options.spin_budget_us)}),
      weight_bytes_(sum_weight_bytes(weights_, cfg_)),
      arena_(cfg_.activation_bytes(max_seq_, max_ctx_, static_cast<std::size_t>(pool_.size()),
                                   cache_type_)) {}

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
  auto table = kernels::select(options.kernels);
  if (!table) {
    return table.error();
  }
  return std::unique_ptr<LlamaModel>(new LlamaModel(std::move(*file), std::move(*cfg),
                                                    std::move(*tokenizer), std::move(*weights),
                                                    options, *table));
}

Expected<std::span<const float>> LlamaModel::forward(std::span<const std::int32_t> tokens,
                                                     const ForwardOptions& opts) {
  return run(tokens, nullptr, opts);
}

Expected<std::span<const float>> LlamaModel::forward_prefill(KVCache& cache,
                                                             std::span<const std::int32_t> tokens,
                                                             const ForwardOptions& opts) {
  return run(tokens, &cache, opts);
}

const char* ForwardProfile::name(Slot slot) noexcept {
  switch (slot) {
  case kQuantize:
    return "quantize";
  case kQkv:
    return "qkv";
  case kAttention:
    return "attention";
  case kOutputProj:
    return "wo";
  case kGateUp:
    return "gate_up";
  case kDown:
    return "down";
  case kLogits:
    return "logits";
  case kOther:
    return "other";
  default:
    return "?";
  }
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
    if (cache->type() != cache_type_) {
      return make_error(ErrorCode::InvalidArgument,
                        "forward: the cache type does not match the one the model was loaded with");
    }
    if (n_past + seq > cache->max_ctx()) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("context overflow: {} cached + {} new tokens exceed {}", n_past,
                                    seq, cache->max_ctx()));
    }
  }

  const kernels::KernelTable& k = *kernels_;
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

  // An fp16 cache needs float staging rows for the new keys and values, like the cache-free
  // pass, before they are rounded into the cache.
  const bool half_cache = cache != nullptr && cache->type() == CacheType::F16;
  arena_.reset();
  Activations a =
      allocate_activations(arena_, cfg_, seq, total, cache == nullptr || half_cache, n_threads);

  // Charges the time since the previous tick to a profile slot. With no profile this is one
  // predictable branch per launch.
  using Clock = std::chrono::steady_clock;
  Clock::time_point last_tick = opts.profile != nullptr ? Clock::now() : Clock::time_point{};
  const auto tick = [&](ForwardProfile::Slot slot) {
    if (opts.profile != nullptr) {
      const auto now = Clock::now();
      opts.profile->seconds[slot] += std::chrono::duration<double>(now - last_tick).count();
      last_tick = now;
    }
  };

  // Quantizes the activation rows once per input that feeds quantized weights: inline for a
  // single row (decode), as one launch for prefill. Returns whether the scratch is in use.
  const auto quantize_input = [&](const float* x, const Tensor& w) {
    if (!needs_q8_activations(w, exact_activations_)) {
      return false;
    }
    const auto n_in = static_cast<std::size_t>(w.ne0);
    const std::size_t nb = n_in / kQK8_0;
    if (seq == 1) {
      k.quantize_row_q8_0(x, a.xq.data(), n_in);
    } else {
      pool_.parallel_for(0, seq, 4, [&](std::size_t begin, std::size_t end, int) {
        for (std::size_t r = begin; r < end; ++r) {
          k.quantize_row_q8_0(x + r * n_in, a.xq.data() + r * nb, n_in);
        }
      });
    }
    return true;
  };

  // About eight chunks per thread: enough to balance fast and slow cores, few enough that
  // the atomic claim per chunk is noise next to the columns it covers.
  const auto chunk_for = [&](std::size_t columns) {
    return std::max<std::size_t>(16, columns / (n_threads * 8));
  };

  // Y[seq][n_out] = X[seq][n_in] W^T for one weight, or for several weights that share the
  // input row, as one launch over the union of their output columns. A chunk that spans a
  // boundary runs the kernel once per segment, so every output element is still computed by
  // the same kernel on the same data as it would be alone.
  struct Segment {
    const Tensor* w;
    float* y;
  };
  const auto mm_fused = [&](std::span<const Segment> segments, const float* x,
                            ForwardProfile::Slot slot) {
    const auto n_in = static_cast<std::size_t>(segments[0].w->ne0);
    const std::size_t nb = n_in / kQK8_0;
    tick(ForwardProfile::kOther);
    bool quantized = false;
    for (const auto& seg : segments) {
      AXON_DCHECK(static_cast<std::size_t>(seg.w->ne0) == n_in,
                  "fused matmul: input widths differ");
      quantized = quantize_input(x, *seg.w) || quantized;
      if (quantized) {
        break;  // one input, quantized once
      }
    }
    if (quantized) {
      tick(ForwardProfile::kQuantize);
    }
    std::size_t total_columns = 0;
    for (const auto& seg : segments) {
      total_columns += static_cast<std::size_t>(seg.w->ne1);
    }
    pool_.parallel_for(
        0, total_columns, chunk_for(total_columns), [&](std::size_t begin, std::size_t end, int) {
          std::size_t offset = 0;
          for (const auto& seg : segments) {
            const auto n_out = static_cast<std::size_t>(seg.w->ne1);
            const std::size_t lo = std::max(begin, offset);
            const std::size_t hi = std::min(end, offset + n_out);
            if (lo < hi) {
              for (std::size_t r = 0; r < seq; ++r) {
                matmul_columns(k, *seg.w, x + r * n_in, quantized ? a.xq.data() + r * nb : nullptr,
                               seg.y + r * n_out, lo - offset, hi - offset, exact_activations_);
              }
            }
            offset += n_out;
          }
        });
    tick(slot);
  };
  const auto mm = [&](const Tensor& w, const float* x, float* y, ForwardProfile::Slot slot) {
    const Segment one[] = {{.w = &w, .y = y}};
    mm_fused(one, x, slot);
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
    // With an fp16 cache the new rows are staged in float and rounded in after RoPE.
    const bool direct = cache != nullptr && !half_cache;
    float* k_base = direct ? cache->k(l) : a.k.data();
    float* v_base = direct ? cache->v(l) : a.v.data();
    float* k_new = direct ? k_base + n_past * kv_row : a.k.data();
    float* v_new = direct ? v_base + n_past * kv_row : a.v.data();
    AXON_DUMP(dump_this, "x_in", a.x);

    rmsnorm_rows(k, a.x, w.attn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "attn_norm_out", a.h);
    {
      const Segment qkv[] = {
          {.w = &w.wq, .y = a.q.data()}, {.w = &w.wk, .y = k_new}, {.w = &w.wv, .y = v_new}};
      mm_fused(qkv, a.h.data(), ForwardProfile::kQkv);
    }
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
    tick(ForwardProfile::kOther);
    if (half_cache) {
      std::uint16_t* k16 = cache->k16(l) + n_past * kv_row;
      std::uint16_t* v16 = cache->v16(l) + n_past * kv_row;
      for (std::size_t i = 0; i < seq * kv_row; ++i) {
        k16[i] = fp32_to_fp16(k_new[i]);
        v16[i] = fp32_to_fp16(v_new[i]);
      }
      tick(ForwardProfile::kOther);
      pool_.parallel_for(0, n_head, 1, [&](std::size_t begin, std::size_t end, int tid) {
        float* scores = a.scores.data() + static_cast<std::size_t>(tid) * seq * total;
        k.attention_kv_heads_f16(a.q.data(), cache->k16(l), cache->v16(l), a.attn.data(), scores,
                                 seq, n_past, n_head, n_head_kv, head_dim, scale, begin, end);
      });
    } else {
      pool_.parallel_for(0, n_head, 1, [&](std::size_t begin, std::size_t end, int tid) {
        float* scores = a.scores.data() + static_cast<std::size_t>(tid) * seq * total;
        k.attention_kv_heads(a.q.data(), k_base, v_base, a.attn.data(), scores, seq, n_past, n_head,
                             n_head_kv, head_dim, scale, begin, end);
      });
    }
    tick(ForwardProfile::kAttention);
    AXON_DUMP(dump_this, "attn_out", a.attn);
    mm(w.wo, a.attn.data(), a.h.data(), ForwardProfile::kOutputProj);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_mid", a.x);

    rmsnorm_rows(k, a.x, w.ffn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "ffn_norm_out", a.h);
    {
      // Gate and up as one launch over column pairs, with the SiLU product applied by the
      // thread that just produced both halves of its chunk.
      tick(ForwardProfile::kOther);
      const bool quantized = quantize_input(a.h.data(), w.w_gate);
      if (quantized) {
        tick(ForwardProfile::kQuantize);
      }
      const std::size_t nb = n_embd / kQK8_0;
      pool_.parallel_for(0, n_ff, chunk_for(n_ff), [&](std::size_t begin, std::size_t end, int) {
        for (std::size_t r = 0; r < seq; ++r) {
          const float* x = a.h.data() + r * n_embd;
          const BlockQ8_0* xq = quantized ? a.xq.data() + r * nb : nullptr;
          float* gate = a.gate.data() + r * n_ff;
          float* up = a.up.data() + r * n_ff;
          matmul_columns(k, w.w_gate, x, xq, gate, begin, end, exact_activations_);
          matmul_columns(k, w.w_up, x, xq, up, begin, end, exact_activations_);
          k.silu_mul(gate + begin, up + begin, gate + begin, end - begin);
        }
      });
      tick(ForwardProfile::kGateUp);
    }
    mm(w.w_down, a.gate.data(), a.h.data(), ForwardProfile::kDown);
    AXON_DUMP(dump_this, "ffn_out", a.h);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_out", a.x);
  }

  rmsnorm_rows(k, a.x, weights_.output_norm, a.h, seq, cfg_.rms_eps);
  mm(weights_.output, a.h.data(), a.logits.data(), ForwardProfile::kLogits);
  AXON_DUMP(dumping, "logits", a.logits);
#undef AXON_DUMP
  tick(ForwardProfile::kOther);
  if (cache != nullptr) {
    cache->advance(seq);
  }
  return std::span<const float>(a.logits);
}

}  // namespace axon::model
