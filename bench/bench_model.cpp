#include "model/llama.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

// Forward wall time per token over a full sequence, on the scalar single-threaded path. The
// synthetic model always runs; SmolLM2 runs when AXON_MODELS_DIR holds the F16 file.
void bm_forward(benchmark::State& state, const std::string& path) {
  const auto seq = static_cast<std::size_t>(state.range(0));
  auto model = axon::model::LlamaModel::load(path, seq);
  if (!model) {
    state.SkipWithError(model.error().message);
    return;
  }
  std::vector<std::int32_t> tokens(seq);
  const auto n_vocab = static_cast<std::int32_t>((*model)->config().n_vocab);
  for (std::size_t i = 0; i < seq; ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 7919) % static_cast<std::size_t>(n_vocab));
  }
  for (auto _ : state) {
    const auto logits = (*model)->forward(tokens);
    benchmark::DoNotOptimize(logits->data());
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(seq));
  state.counters["arena_MB"] =
      static_cast<double>((*model)->arena_high_water()) / (1024.0 * 1024.0);
}

std::string synthetic_path() {
  return AXON_FIXTURE_DIR "/gguf/synthetic-2l.gguf";
}

std::string smollm2_path() {
  const char* dir = std::getenv("AXON_MODELS_DIR");
  if (dir == nullptr) {
    return {};
  }
  const auto path = std::filesystem::path(dir) / "SmolLM2-135M-Instruct-f16.gguf";
  return std::filesystem::exists(path) ? path.string() : std::string{};
}

const int kRegistered = [] {
  benchmark::RegisterBenchmark("bm_forward_synthetic", bm_forward, synthetic_path())
      ->Arg(1)
      ->Arg(32)
      ->Arg(128)
      ->Unit(benchmark::kMillisecond);
  if (const auto path = smollm2_path(); !path.empty()) {
    benchmark::RegisterBenchmark("bm_forward_smollm2_f16", bm_forward, path)
        ->Arg(1)
        ->Arg(32)
        ->Arg(128)
        ->Unit(benchmark::kMillisecond)
        ->MinTime(0.5);
  }
  return 0;
}();

}  // namespace
