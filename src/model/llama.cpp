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

// Y[n_rows][n_out] = X[n_rows][n_in] W^T over output columns [begin, end) of the weight, the
// activation rows n_in apart and the output rows w.ne1 apart. Quantized weights take the rows'
// Q8_0 blocks in `xq` unless `exact` asks for the float path. One row goes to the single-row
// kernel, more rows to the multi-row entry with the output stride. Each output element is
// computed by exactly one call with the same instruction sequence whatever the split, which is
// what keeps threaded results bit-identical.
void matmul_columns(const kernels::KernelTable& k, const Tensor& w, const float* x,
                    const BlockQ8_0* xq, float* y, std::size_t n_rows, std::size_t begin,
                    std::size_t end, bool exact) {
  const auto n_in = static_cast<std::size_t>(w.ne0);
  const auto stride = static_cast<std::size_t>(w.ne1);
  const std::size_t n_out = end - begin;
  const auto* base = static_cast<const std::byte*>(w.data) + begin * w.row_bytes;
  float* out = y + begin;
  const bool one = n_rows == 1;
  switch (w.type) {
  case DType::F32: {
    const auto* wp = reinterpret_cast<const float*>(base);
    one ? k.matmul_f32(wp, x, out, n_out, n_in)
        : k.matmul_f32_rows(wp, x, out, n_rows, n_out, n_in, stride);
    return;
  }
  case DType::F16: {
    const auto* wp = reinterpret_cast<const std::uint16_t*>(base);
    one ? k.matmul_f16(wp, x, out, n_out, n_in)
        : k.matmul_f16_rows(wp, x, out, n_rows, n_out, n_in, stride);
    return;
  }
  case DType::BF16: {
    const auto* wp = reinterpret_cast<const std::uint16_t*>(base);
    one ? k.matmul_bf16(wp, x, out, n_out, n_in)
        : k.matmul_bf16_rows(wp, x, out, n_rows, n_out, n_in, stride);
    return;
  }
  case DType::Q8_0: {
    const auto* wp = reinterpret_cast<const BlockQ8_0*>(base);
    if (exact) {
      one ? k.matmul_q8_0(wp, x, out, n_out, n_in)
          : k.matmul_q8_0_rows(wp, x, out, n_rows, n_out, n_in, stride);
    } else {
      one ? k.matmul_q8_0_q8(wp, xq, out, n_out, n_in)
          : k.matmul_q8_0_q8_rows(wp, xq, out, n_rows, n_out, n_in, stride);
    }
    return;
  }
  case DType::Q4_0: {
    const auto* wp = reinterpret_cast<const BlockQ4_0*>(base);
    if (exact) {
      one ? k.matmul_q4_0(wp, x, out, n_out, n_in)
          : k.matmul_q4_0_rows(wp, x, out, n_rows, n_out, n_in, stride);
    } else {
      one ? k.matmul_q4_0_q8(wp, xq, out, n_out, n_in)
          : k.matmul_q4_0_q8_rows(wp, xq, out, n_rows, n_out, n_in, stride);
    }
    return;
  }
  case DType::Q4_1: {
    const auto* wp = reinterpret_cast<const BlockQ4_1*>(base);
    if (exact) {
      one ? k.matmul_q4_1(wp, x, out, n_out, n_in)
          : k.matmul_q4_1_rows(wp, x, out, n_rows, n_out, n_in, stride);
    } else {
      one ? k.matmul_q4_1_q8(wp, xq, out, n_out, n_in)
          : k.matmul_q4_1_q8_rows(wp, xq, out, n_rows, n_out, n_in, stride);
    }
    return;
  }
  case DType::Q6_K: {
    // Float activations only: the super-block format has no integer path yet.
    const auto* wp = reinterpret_cast<const BlockQ6_K*>(base);
    one ? k.matmul_q6_k(wp, x, out, n_out, n_in)
        : k.matmul_q6_k_rows(wp, x, out, n_rows, n_out, n_in, stride);
    return;
  }
  default:
    AXON_CHECK(false, "matmul: unbound dtype {}", dtype_name(w.type));
  }
}

// Query rows per attention work item. Sixteen rows of a 128-token prefill chunk give eight
// groups per head, enough items to balance four or eight threads over nine heads.
constexpr std::size_t kAttentionRows = 16;

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
      cache_type_(options.cache), max_batch_(std::max<std::size_t>(1, options.max_batch)),
      exact_activations_(options.exact_activations), kernels_(table),
      pool_(ThreadPoolOptions{.n_threads = options.n_threads,
                              .spin_budget = std::chrono::microseconds(options.spin_budget_us)}),
      weight_bytes_(sum_weight_bytes(weights_, cfg_)),
      arena_(cfg_.activation_bytes(max_seq_, max_ctx_, static_cast<std::size_t>(pool_.size()),
                                   cache_type_, max_batch_)) {}

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
  case Quantize:
    return "quantize";
  case Qkv:
    return "qkv";
  case Attention:
    return "attention";
  case OutputProj:
    return "wo";
  case GateUp:
    return "gate_up";
  case Down:
    return "down";
  case Logits:
    return "logits";
  case Other:
    return "other";
  default:
    return "?";
  }
}

Expected<std::span<const float>> LlamaModel::forward_one(KVCache& cache, std::int32_t token) {
  return run(std::span<const std::int32_t>(&token, 1), &cache, ForwardOptions{});
}

Expected<std::span<const float>> LlamaModel::forward_batch(std::span<KVCache* const> caches,
                                                           std::span<const std::int32_t> tokens,
                                                           const ForwardOptions& opts) {
  if (caches.empty() || caches.size() != tokens.size()) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("forward_batch: {} caches for {} tokens", caches.size(), tokens.size()));
  }
  if (caches.size() > max_batch_) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("forward_batch: {} sequences exceed max_batch {}", caches.size(), max_batch_));
  }
  for (std::size_t i = 0; i < caches.size(); ++i) {
    const KVCache* cache = caches[i];
    if (cache == nullptr || !cache->matches(cfg_)) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("forward_batch: cache {} was built for a different model", i));
    }
    if (cache->type() != cache_type_) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("forward_batch: cache {} has a different type from the one "
                                    "the model was loaded with",
                                    i));
    }
    if (cache->pos() >= cache->max_ctx()) {
      return make_error(ErrorCode::InvalidArgument,
                        fmt::format("context overflow: cache {} holds {} of {} positions", i,
                                    cache->pos(), cache->max_ctx()));
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (caches[j] == cache) {
        return make_error(ErrorCode::InvalidArgument,
                          fmt::format("forward_batch: cache {} appears twice", i));
      }
    }
  }
  return run(tokens, nullptr, opts, caches);
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
                                                 KVCache* cache, const ForwardOptions& opts,
                                                 std::span<KVCache* const> batch) {
  // Batched: every row is its own sequence with its own cache and position, validated by
  // forward_batch. The rows share the matmuls like a prefill chunk does and part ways only
  // where a position or a cache is read or written.
  const bool batched = !batch.empty();
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
  const auto pos_of = [&](std::size_t i) { return batched ? batch[i]->pos() : n_past + i; };
  std::size_t total = n_past + seq;
  if (batched) {
    total = 0;
    for (const KVCache* c : batch) {
      total = std::max(total, c->pos() + 1);
    }
  }
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
  const bool half_cache = (cache != nullptr && cache->type() == CacheType::F16) ||
                          (batched && cache_type_ == CacheType::F16);
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
    tick(ForwardProfile::Other);
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
      tick(ForwardProfile::Quantize);
    }
    std::size_t total_columns = 0;
    for (const auto& seg : segments) {
      total_columns += static_cast<std::size_t>(seg.w->ne1);
    }
    pool_.parallel_for(0, total_columns, chunk_for(total_columns),
                       [&](std::size_t begin, std::size_t end, int) {
                         std::size_t offset = 0;
                         for (const auto& seg : segments) {
                           const auto n_out = static_cast<std::size_t>(seg.w->ne1);
                           const std::size_t lo = std::max(begin, offset);
                           const std::size_t hi = std::min(end, offset + n_out);
                           if (lo < hi) {
                             matmul_columns(k, *seg.w, x, quantized ? a.xq.data() : nullptr, seg.y,
                                            seq, lo - offset, hi - offset, exact_activations_);
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
      mm_fused(qkv, a.h.data(), ForwardProfile::Qkv);
    }
    if (w.bq && w.bk && w.bv) {
      add_bias(a.q.data(), *w.bq, seq);
      add_bias(k_new, *w.bk, seq);
      add_bias(v_new, *w.bv, seq);
    }
    for (std::size_t i = 0; i < seq; ++i) {
      k.rope(a.q.data() + i * n_head * head_dim, n_head, pos_of(i), rope);
      k.rope(k_new + i * kv_row, n_head_kv, pos_of(i), rope);
    }
    AXON_DUMP(dump_this, "q", a.q);
    AXON_DUMP(dump_this, "k", std::span<const float>(k_new, seq * kv_row));
    AXON_DUMP(dump_this, "v", std::span<const float>(v_new, seq * kv_row));
    tick(ForwardProfile::Other);
    if (batched) {
      // Each row's key and value go to its own cache at its own position, then every
      // (sequence, head) pair is one work item over that sequence's cache.
      for (std::size_t i = 0; i < seq; ++i) {
        KVCache* c = batch[i];
        const std::size_t at = c->pos() * kv_row;
        if (half_cache) {
          std::uint16_t* k16 = c->k16(l) + at;
          std::uint16_t* v16 = c->v16(l) + at;
          for (std::size_t j = 0; j < kv_row; ++j) {
            k16[j] = fp32_to_fp16(k_new[i * kv_row + j]);
            v16[j] = fp32_to_fp16(v_new[i * kv_row + j]);
          }
        } else {
          std::copy_n(k_new + i * kv_row, kv_row, c->k(l) + at);
          std::copy_n(v_new + i * kv_row, kv_row, c->v(l) + at);
        }
      }
      tick(ForwardProfile::Other);
      const std::size_t q_row = n_head * head_dim;
      pool_.parallel_for(0, seq * n_head, 1, [&](std::size_t begin, std::size_t end, int tid) {
        float* scores = a.scores.data() + static_cast<std::size_t>(tid) * seq * total;
        for (std::size_t item = begin; item < end; ++item) {
          const std::size_t b = item / n_head;
          const std::size_t h = item % n_head;
          KVCache* c = batch[b];
          if (half_cache) {
            k.attention_kv_heads_f16(a.q.data() + b * q_row, c->k16(l), c->v16(l),
                                     a.attn.data() + b * q_row, scores, 1, c->pos(), n_head,
                                     n_head_kv, head_dim, scale, h, h + 1);
          } else {
            k.attention_kv_heads(a.q.data() + b * q_row, c->k(l), c->v(l),
                                 a.attn.data() + b * q_row, scores, 1, c->pos(), n_head, n_head_kv,
                                 head_dim, scale, h, h + 1);
          }
        }
      });
    } else if (half_cache) {
      std::uint16_t* k16 = cache->k16(l) + n_past * kv_row;
      std::uint16_t* v16 = cache->v16(l) + n_past * kv_row;
      for (std::size_t i = 0; i < seq * kv_row; ++i) {
        k16[i] = fp32_to_fp16(k_new[i]);
        v16[i] = fp32_to_fp16(v_new[i]);
      }
      tick(ForwardProfile::Other);
    }
    if (!batched) {
      // One work item per head and group of query rows, so a prompt spreads over more items
      // than there are heads. Each item is a plain call on its rows: the row offset moves the
      // query and output pointers and the past length, and a thread's score scratch, sized for
      // the whole call, holds any group. Which thread runs a row changes, how it is computed
      // does not, so the logits stay bit-identical at any thread count.
      const std::size_t n_groups = (seq + kAttentionRows - 1) / kAttentionRows;
      const std::size_t q_row = n_head * head_dim;
      pool_.parallel_for(0, n_head * n_groups, 1, [&](std::size_t begin, std::size_t end, int tid) {
        float* scores = a.scores.data() + static_cast<std::size_t>(tid) * seq * total;
        for (std::size_t item = begin; item < end; ++item) {
          const std::size_t h = item % n_head;
          const std::size_t first = (item / n_head) * kAttentionRows;
          const std::size_t rows = std::min(kAttentionRows, seq - first);
          const float* q_rows = a.q.data() + first * q_row;
          float* out_rows = a.attn.data() + first * q_row;
          if (half_cache) {
            k.attention_kv_heads_f16(q_rows, cache->k16(l), cache->v16(l), out_rows, scores, rows,
                                     n_past + first, n_head, n_head_kv, head_dim, scale, h, h + 1);
          } else {
            k.attention_kv_heads(q_rows, k_base, v_base, out_rows, scores, rows, n_past + first,
                                 n_head, n_head_kv, head_dim, scale, h, h + 1);
          }
        }
      });
    }
    tick(ForwardProfile::Attention);
    AXON_DUMP(dump_this, "attn_out", a.attn);
    mm(w.wo, a.attn.data(), a.h.data(), ForwardProfile::OutputProj);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_mid", a.x);

    rmsnorm_rows(k, a.x, w.ffn_norm, a.h, seq, cfg_.rms_eps);
    AXON_DUMP(dump_this, "ffn_norm_out", a.h);
    {
      // Gate and up as one launch over column pairs, with the SiLU product applied by the
      // thread that just produced both halves of its chunk.
      tick(ForwardProfile::Other);
      const bool quantized = quantize_input(a.h.data(), w.w_gate);
      if (quantized) {
        tick(ForwardProfile::Quantize);
      }
      pool_.parallel_for(0, n_ff, chunk_for(n_ff), [&](std::size_t begin, std::size_t end, int) {
        const BlockQ8_0* xq = quantized ? a.xq.data() : nullptr;
        matmul_columns(k, w.w_gate, a.h.data(), xq, a.gate.data(), seq, begin, end,
                       exact_activations_);
        matmul_columns(k, w.w_up, a.h.data(), xq, a.up.data(), seq, begin, end, exact_activations_);
        for (std::size_t r = 0; r < seq; ++r) {
          float* gate = a.gate.data() + r * n_ff + begin;
          const float* up = a.up.data() + r * n_ff + begin;
          k.silu_mul(gate, up, gate, end - begin);
        }
      });
      tick(ForwardProfile::GateUp);
    }
    mm(w.w_down, a.gate.data(), a.h.data(), ForwardProfile::Down);
    AXON_DUMP(dump_this, "ffn_out", a.h);
    add_rows(a.x, a.h);
    AXON_DUMP(dump_this, "x_out", a.x);
  }

  rmsnorm_rows(k, a.x, weights_.output_norm, a.h, seq, cfg_.rms_eps);
  mm(weights_.output, a.h.data(), a.logits.data(), ForwardProfile::Logits);
  AXON_DUMP(dumping, "logits", a.logits);
#undef AXON_DUMP
  tick(ForwardProfile::Other);
  if (cache != nullptr) {
    cache->advance(seq);
  }
  for (KVCache* c : batch) {
    c->advance(1);
  }
  return std::span<const float>(a.logits);
}

}  // namespace axon::model
