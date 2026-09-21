#include "axon/tokenizer.h"

#include "engine/perplexity.h"
#include "model/llama.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

namespace {

struct Options {
  std::string model;
  std::string text = AXON_PERPLEXITY_TEXT;
  std::size_t chunks = 0;
  std::size_t chunk_size = 512;
  std::size_t max_seq = 128;
  bool exact_activations = false;
  bool json = false;
};

int run(int argc, char** argv) {
  CLI::App app{"axon-perplexity: perplexity of a GGUF model over a text, scored in chunks"};
  Options opt;
  app.add_option("--model,-m", opt.model, "path to a .gguf file")->required();
  app.add_option("--text", opt.text, "UTF-8 text file (default: the committed WikiText-2 head)");
  app.add_option("--chunks", opt.chunks, "score at most this many chunks, 0 for all")
      ->default_val(0);
  app.add_option("--chunk-size", opt.chunk_size, "tokens per chunk, the second half is scored")
      ->default_val(512);
  app.add_option("--max-seq", opt.max_seq, "tokens per forward call")->default_val(128);
  app.add_flag("--exact-activations", opt.exact_activations,
               "keep activations in float for quantized weights instead of Q8_0 blocks");
  app.add_flag("--json", opt.json, "print one JSON object instead of text");
  CLI11_PARSE(app, argc, argv);

  std::ifstream in(opt.text, std::ios::binary);
  if (!in) {
    fmt::print(stderr, "error: cannot open {}\n", opt.text);
    return 2;
  }
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

  const auto t0 = std::chrono::steady_clock::now();
  auto loaded =
      axon::model::LlamaModel::load(opt.model, {.max_seq = std::min(opt.max_seq, opt.chunk_size),
                                                .max_ctx = opt.chunk_size,
                                                .exact_activations = opt.exact_activations});
  if (!loaded) {
    fmt::print(stderr, "error: {}\n", loaded.error().message);
    return 2;
  }
  auto& model = **loaded;
  const auto tokens =
      model.tokenizer().encode(text, {.add_bos = std::nullopt, .parse_special = false});
  if (!opt.json) {
    fmt::print("model: {} ({})\n", model.file().model_name(), model.config().describe());
    fmt::print("text: {} bytes, {} tokens, chunks of {}\n", text.size(), tokens.size(),
               opt.chunk_size);
  }
  const auto result = axon::engine::perplexity(
      model, tokens, {.chunk_size = opt.chunk_size, .max_chunks = opt.chunks},
      [&](std::size_t chunk, double running) {
        if (!opt.json) {
          fmt::print("[{}] {:.4f}\n", chunk + 1, running);
          std::fflush(stdout);
        }
      });
  if (!result) {
    fmt::print(stderr, "error: {}\n", result.error().message);
    return 2;
  }
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (opt.json) {
    fmt::print(
        "{{\"model\": \"{}\", \"perplexity\": {:.6f}, \"nll_sum\": {:.6f}, \"scored_tokens\": {}, "
        "\"chunks\": {}, \"chunk_size\": {}, \"exact_activations\": {}, \"seconds\": {:.2f}}}\n",
        opt.model, result->perplexity(), result->nll_sum, result->scored_tokens, result->chunks,
        opt.chunk_size, opt.exact_activations, seconds);
  } else {
    fmt::print("perplexity {:.4f} over {} scored tokens in {} chunks, {:.1f} s\n",
               result->perplexity(), result->scored_tokens, result->chunks, seconds);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
}
