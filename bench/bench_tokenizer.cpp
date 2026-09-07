#include "axon/tokenizer.h"
#include "core/gguf.h"

#include <benchmark/benchmark.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// Encode throughput on the corpus, repeated to a few hundred KB. Reported as bytes/s.
void bm_tokenizer_encode(benchmark::State& state, const char* fixture) {
  const auto file = axon::GGUFFile::open(std::string(AXON_FIXTURE_DIR) + "/tokenizer/" + fixture);
  if (!file) {
    state.SkipWithError(file.error().message.c_str());
    return;
  }
  auto tokenizer = axon::Tokenizer::from_gguf(*file);
  if (!tokenizer) {
    state.SkipWithError(tokenizer.error().message.c_str());
    return;
  }
  std::string text;
  const std::string corpus = read_file(std::string(AXON_FIXTURE_DIR) + "/../tokenizer_corpus.txt");
  while (text.size() < std::size_t{256} * 1024) {
    text += corpus;
  }
  std::size_t tokens = 0;
  for (auto _ : state) {
    const auto ids = (*tokenizer)->encode(text, {.add_bos = false});
    tokens = ids.size();
    benchmark::DoNotOptimize(ids.data());
  }
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(text.size()));
  state.counters["tokens"] = static_cast<double>(tokens);
}

}  // namespace

BENCHMARK_CAPTURE(bm_tokenizer_encode, smollm2_bpe, "smollm2.tokenizer.gguf")
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(bm_tokenizer_encode, tinyllama_spm, "tinyllama.tokenizer.gguf")
    ->Unit(benchmark::kMillisecond);
