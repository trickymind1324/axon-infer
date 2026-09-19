#include "axon/tokenizer.h"
#include "axon/version.h"
#include "core/cpu_info.h"
#include "core/gguf.h"
#include "core/log.h"

#include "model/llama.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

std::string compiler_string() {
#if defined(__clang__)
  return fmt::format("clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  return fmt::format("gcc {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

void print_version() {
  fmt::print("axon-infer {} (commit {})\n", axon::version(), axon::git_commit());
  fmt::print("compiler: {}\n", compiler_string());
  fmt::print("cpu: {}\n", axon::cpu_info().summary());
}

struct TokenizeOptions {
  std::string model;
  std::string text;
  bool no_bos = false;
  bool no_special = false;
  bool ids_only = false;
};

std::string printable(std::string_view bytes) {
  std::string out;
  for (const char c : bytes) {
    switch (c) {
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    case '\r':
      out += "\\r";
      break;
    case '"':
      out += "\\\"";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        out += fmt::format("\\x{:02x}", static_cast<unsigned char>(c));
      } else {
        out.push_back(c);
      }
    }
  }
  return out;
}

int run_tokenize(const TokenizeOptions& opt) {
  auto file = axon::GGUFFile::open(opt.model);
  if (!file) {
    fmt::print(stderr, "error: {}\n", file.error().message);
    return 2;
  }
  auto tokenizer = axon::Tokenizer::from_gguf(*file);
  if (!tokenizer) {
    fmt::print(stderr, "error: {}\n", tokenizer.error().message);
    return 2;
  }
  std::string text = opt.text;
  if (text.empty()) {
    text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
  }
  const auto ids =
      (*tokenizer)
          ->encode(text, {.add_bos = opt.no_bos ? std::optional<bool>(false) : std::nullopt,
                          .parse_special = !opt.no_special});
  if (opt.ids_only) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
      fmt::print("{}{}", i == 0 ? "" : " ", ids[i]);
    }
    fmt::print("\n");
    return 0;
  }
  fmt::print("model: {} ({}, pre {}), vocab {}\n", (*file).model_name(), (*tokenizer)->model(),
             (*tokenizer)->pre(), (*tokenizer)->vocab_size());
  for (const std::int32_t id : ids) {
    fmt::print("[{:>7}] \"{}\"{}\n", id, printable((*tokenizer)->token_bytes(id)),
               (*tokenizer)->is_special(id) ? "  (special)" : "");
  }
  fmt::print("{} tokens for {} bytes\n", ids.size(), text.size());
  return 0;
}

struct GenerateOptions {
  std::string model;
  std::string prompt;
  int n_tokens = 20;
  std::size_t max_seq = 256;
  std::size_t ctx = 2048;
  bool no_cache = false;
  bool no_bos = false;
  std::size_t dump_layer = axon::model::ForwardOptions::kNoDump;
  std::string dump_dir = "dump-axon";
};

// Greedy generation. The default path prefills the prompt into a KV cache and decodes one
// token per step. --no-cache keeps the cache-free loop that recomputes the whole sequence each
// step, which is also the only path that can dump activations.
int run_generate(const GenerateOptions& opt) {
  const bool dumping = opt.dump_layer != axon::model::ForwardOptions::kNoDump;
  const bool use_cache = !opt.no_cache && !dumping;
  const auto t_load = std::chrono::steady_clock::now();
  const std::size_t ctx = use_cache ? opt.ctx : opt.max_seq;
  auto loaded = axon::model::LlamaModel::load(
      opt.model, {.max_seq = std::min(opt.max_seq, ctx), .max_ctx = ctx});
  if (!loaded) {
    fmt::print(stderr, "error: {}\n", loaded.error().message);
    return 2;
  }
  auto& model = **loaded;
  const auto& tokenizer = model.tokenizer();
  std::string prompt = opt.prompt;
  if (prompt.empty()) {
    prompt.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
  }
  std::vector<std::int32_t> ids =
      tokenizer.encode(prompt, {.add_bos = opt.no_bos ? std::optional<bool>(false) : std::nullopt});
  fmt::print("model: {} ({})\n", model.file().model_name(), model.config().describe());
  fmt::print("prompt: {} tokens\n", ids.size());
  const std::size_t limit = use_cache ? model.max_ctx() : model.max_seq();
  if (ids.size() >= limit) {
    fmt::print(stderr, "error: prompt has {} tokens, which leaves no room in a context of {}\n",
               ids.size(), limit);
    return 2;
  }
  const auto load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

  axon::model::ForwardOptions forward_options;
  if (dumping) {
    forward_options.dump_layer = opt.dump_layer;
    forward_options.dump_dir = opt.dump_dir;
    std::filesystem::create_directories(forward_options.dump_dir);
  }
  const std::size_t n_vocab = model.config().n_vocab;
  const auto pick = [&](std::span<const float> logits) {
    const float* last = logits.data() + (logits.size() / n_vocab - 1) * n_vocab;
    return static_cast<std::int32_t>(std::ranges::max_element(last, last + n_vocab) - last);
  };
  const auto eog = tokenizer.eog_tokens();
  const auto is_eog = [&](std::int32_t id) { return std::ranges::find(eog, id) != eog.end(); };

  fmt::print("{}", prompt);
  std::fflush(stdout);
  axon::StreamingDecoder decoder(tokenizer);
  std::optional<axon::model::KVCache> cache;
  if (use_cache) {
    cache.emplace(model.new_cache());
  }

  // Prefill: the prompt in chunks of max_seq with a cache, or one dump-capable pass without.
  const auto t_prefill = std::chrono::steady_clock::now();
  std::int32_t next = 0;
  if (use_cache) {
    for (std::size_t offset = 0; offset < ids.size(); offset += model.max_seq()) {
      const std::size_t n = std::min(model.max_seq(), ids.size() - offset);
      const auto logits = model.forward_prefill(*cache, std::span(ids).subspan(offset, n));
      if (!logits) {
        fmt::print(stderr, "\nerror: {}\n", logits.error().message);
        return 2;
      }
      next = pick(*logits);
    }
  } else {
    const auto logits = model.forward(ids, forward_options);
    if (!logits) {
      fmt::print(stderr, "\nerror: {}\n", logits.error().message);
      return 2;
    }
    next = pick(*logits);
  }
  const auto prefill_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill).count();

  const auto t_decode = std::chrono::steady_clock::now();
  int generated = 0;
  while (generated < opt.n_tokens && !is_eog(next) && ids.size() < limit) {
    ids.push_back(next);
    ++generated;
    fmt::print("{}", decoder.push(next));
    std::fflush(stdout);
    if (generated == opt.n_tokens || ids.size() >= limit) {
      break;
    }
    const auto logits = use_cache ? model.forward_one(*cache, next) : model.forward(ids);
    if (!logits) {
      fmt::print(stderr, "\nerror: {}\n", logits.error().message);
      return 2;
    }
    next = pick(*logits);
  }
  fmt::print("{}\n", decoder.flush());
  const auto decode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode).count();
  const auto rate = [](int n, double seconds) {
    return seconds > 0.0 && n > 0 ? static_cast<double>(n) / seconds : 0.0;
  };
  const auto prompt_tokens = static_cast<int>(ids.size()) - generated;
  fmt::print("prefill {} tokens in {:.2f} s ({:.1f} tok/s), decode {} tokens in {:.2f} s "
             "({:.2f} tok/s), load {:.2f} s, arena {:.1f} MB",
             prompt_tokens, prefill_seconds, rate(prompt_tokens, prefill_seconds), generated,
             decode_seconds, rate(generated, decode_seconds), load_seconds,
             static_cast<double>(model.arena_high_water()) / (1024.0 * 1024.0));
  if (cache) {
    fmt::print(", cache {:.1f} MB of {:.1f} MB",
               static_cast<double>(cache->live_bytes()) / (1024.0 * 1024.0),
               static_cast<double>(cache->bytes()) / (1024.0 * 1024.0));
  } else {
    fmt::print(", no cache");
  }
  fmt::print("\n");
  if (dumping) {
    fmt::print("layer {} activations written to {}\n", opt.dump_layer, opt.dump_dir);
  }
  return 0;
}

int run(int argc, char** argv) {
  CLI::App app{"axon-infer: LLM inference engine in C++20"};
  app.set_version_flag("--version", [] {
    print_version();
    return std::string{};
  });

  std::string log_level = "info";
  app.add_option("--log-level", log_level, "trace|debug|info|warn|error|off")->default_str("info");

  TokenizeOptions tokenize;
  auto* tok =
      app.add_subcommand("tokenize", "encode text with a model's tokenizer and print the tokens");
  tok->add_option("--model,-m", tokenize.model, "path to a .gguf file")->required();
  tok->add_option("text", tokenize.text, "text to encode (reads stdin when omitted)");
  tok->add_flag("--no-bos", tokenize.no_bos, "do not prepend the BOS token");
  tok->add_flag("--no-special", tokenize.no_special,
                "treat special tokens in the text as plain text");
  tok->add_flag("--ids-only", tokenize.ids_only, "print only the ids on one line");

  GenerateOptions generate;
  auto* gen = app.add_subcommand("generate", "greedy text generation (full recompute, no cache)");
  gen->add_option("--model,-m", generate.model, "path to a .gguf file")->required();
  gen->add_option("--prompt,-p", generate.prompt, "prompt text (reads stdin when omitted)");
  gen->add_option("--n-tokens,-n", generate.n_tokens, "tokens to generate")->default_val(20);
  gen->add_option("--max-seq", generate.max_seq, "tokens per forward call (prompt chunk size)")
      ->default_val(256);
  gen->add_option("--ctx", generate.ctx, "key/value cache length (prompt plus generated tokens)")
      ->default_val(2048);
  gen->add_flag("--no-cache", generate.no_cache,
                "recompute the whole sequence every step instead of using the cache");
  gen->add_flag("--no-bos", generate.no_bos, "do not prepend the BOS token");
  gen->add_option("--dump-layer", generate.dump_layer,
                  "write this layer's activations from the first forward pass");
  gen->add_option("--dump-dir", generate.dump_dir, "directory for --dump-layer")
      ->default_str("dump-axon");
  app.require_subcommand(0, 1);
  CLI11_PARSE(app, argc, argv);

  axon::log::Level level{};
  if (!axon::log::parse_level(log_level, level)) {
    fmt::print(stderr,
               "error: --log-level: expected one of trace|debug|info|warn|error|off, found '{}'\n",
               log_level);
    return 2;
  }
  axon::log::set_level(level);

  if (tok->parsed()) {
    return run_tokenize(tokenize);
  }
  if (gen->parsed()) {
    return run_generate(generate);
  }
  fmt::print("{}\n", app.help());
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
