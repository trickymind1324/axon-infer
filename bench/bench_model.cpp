#include "model/llama.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

// Wall time per token on the scalar single-threaded path. Prefill runs the whole sequence
// into an empty cache. Decode times one token at a fixed context depth: the cache is filled
// once, then each iteration truncates back and decodes the same position again. The synthetic
// model always runs, SmolLM2 when AXON_MODELS_DIR holds the F16 file.
std::unique_ptr<axon::model::LlamaModel> load_or_skip(benchmark::State& state,
                                                      const std::string& path, std::size_t max_seq,
                                                      std::size_t max_ctx) {
  auto model = axon::model::LlamaModel::load(path, {.max_seq = max_seq, .max_ctx = max_ctx});
  if (!model) {
    state.SkipWithError(model.error().message);
    return nullptr;
  }
  return std::move(*model);
}

std::vector<std::int32_t> filler(std::size_t n, std::int32_t n_vocab) {
  std::vector<std::int32_t> tokens(n);
  for (std::size_t i = 0; i < n; ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 7919) % static_cast<std::size_t>(n_vocab));
  }
  return tokens;
}

void bm_prefill(benchmark::State& state, const std::string& path) {
  const auto seq = static_cast<std::size_t>(state.range(0));
  auto model = load_or_skip(state, path, seq, seq);
  if (!model) {
    return;
  }
  const auto tokens = filler(seq, static_cast<std::int32_t>(model->config().n_vocab));
  auto cache = model->new_cache();
  for (auto _ : state) {
    cache.clear();
    const auto logits = model->forward_prefill(cache, tokens);
    benchmark::DoNotOptimize(logits->data());
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(seq));
  state.counters["arena_MB"] = static_cast<double>(model->arena_high_water()) / (1024.0 * 1024.0);
}

void bm_decode(benchmark::State& state, const std::string& path) {
  const auto context = static_cast<std::size_t>(state.range(0));
  const std::size_t chunk = std::min<std::size_t>(64, context);
  auto model = load_or_skip(state, path, chunk, context);
  if (!model) {
    return;
  }
  if (model->max_ctx() < context) {
    state.SkipWithError("context exceeds the model's training length");
    return;
  }
  const auto tokens = filler(context, static_cast<std::int32_t>(model->config().n_vocab));
  auto cache = model->new_cache();
  for (std::size_t offset = 0; offset + 1 < context; offset += chunk) {
    const std::size_t n = std::min(chunk, context - 1 - offset);
    const auto logits = model->forward_prefill(cache, std::span(tokens).subspan(offset, n));
    if (!logits) {
      state.SkipWithError(logits.error().message);
      return;
    }
  }
  const std::size_t filled = cache.pos();
  for (auto _ : state) {
    cache.truncate(filled);
    const auto logits = model->forward_one(cache, tokens.back());
    benchmark::DoNotOptimize(logits->data());
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["cache_MB"] = static_cast<double>(cache.live_bytes()) / (1024.0 * 1024.0);
}

// One batched decode step over `batch` sequences at the same context, against the single
// step above: the weight pass is shared, the attention is per sequence. Args are {context, batch}.
void bm_decode_batch(benchmark::State& state, const std::string& path) {
  const auto context = static_cast<std::size_t>(state.range(0));
  const auto batch = static_cast<std::size_t>(state.range(1));
  const std::size_t chunk = std::min<std::size_t>(64, context);
  auto model = axon::model::LlamaModel::load(
      path, {.max_seq = chunk, .max_ctx = context, .max_batch = batch});
  if (!model) {
    state.SkipWithError(model.error().message);
    return;
  }
  const auto tokens = filler(context, static_cast<std::int32_t>((*model)->config().n_vocab));
  std::vector<axon::model::KVCache> caches;
  caches.reserve(batch);
  std::vector<axon::model::KVCache*> ptrs;
  for (std::size_t b = 0; b < batch; ++b) {
    caches.push_back((*model)->new_cache());
    for (std::size_t offset = 0; offset + 1 < context; offset += chunk) {
      const std::size_t n = std::min(chunk, context - 1 - offset);
      const auto logits =
          (*model)->forward_prefill(caches.back(), std::span(tokens).subspan(offset, n));
      if (!logits) {
        state.SkipWithError(logits.error().message);
        return;
      }
    }
    ptrs.push_back(&caches.back());
  }
  const std::size_t filled = caches.front().pos();
  const std::vector<std::int32_t> row(batch, tokens.back());
  for (auto _ : state) {
    for (auto& c : caches) {
      c.truncate(filled);
    }
    const auto logits = (*model)->forward_batch(ptrs, row);
    benchmark::DoNotOptimize(logits->data());
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
}

constexpr const char* kSyntheticPath = AXON_FIXTURE_DIR "/gguf/synthetic-2l.gguf";

std::string smollm2_path() {
  const char* dir = std::getenv("AXON_MODELS_DIR");
  if (dir == nullptr) {
    return {};
  }
  const auto path = std::filesystem::path(dir) / "SmolLM2-135M-Instruct-f16.gguf";
  return std::filesystem::exists(path) ? path.string() : std::string{};
}

const int kRegistered = [] {
  const std::string synthetic(kSyntheticPath);
  benchmark::RegisterBenchmark("bm_prefill_synthetic", bm_prefill, synthetic)
      ->Arg(32)
      ->Arg(128)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("bm_decode_batch_synthetic", bm_decode_batch, synthetic)
      ->Args({128, 1})
      ->Args({128, 2})
      ->Args({128, 4})
      ->Args({128, 8});
  benchmark::RegisterBenchmark("bm_decode_synthetic", bm_decode, synthetic)
      ->Arg(32)
      ->Arg(128)
      ->Unit(benchmark::kMillisecond);
  if (const auto path = smollm2_path(); !path.empty()) {
    benchmark::RegisterBenchmark("bm_prefill_smollm2_f16", bm_prefill, path)
        ->Arg(32)
        ->Arg(128)
        ->Unit(benchmark::kMillisecond)
        ->MinTime(0.5);
    benchmark::RegisterBenchmark("bm_decode_smollm2_f16", bm_decode, path)
        ->Arg(32)
        ->Arg(512)
        ->Arg(2048)
        ->Unit(benchmark::kMillisecond)
        ->MinTime(0.5);
  }
  return 0;
}();

}  // namespace
