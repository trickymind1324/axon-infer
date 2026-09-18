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
  bool no_bos = false;
  std::size_t dump_layer = axon::model::ForwardOptions::kNoDump;
  std::string dump_dir = "dump-axon";
};

// Greedy generation without a cache: every step runs the whole sequence again. Slow by
// design, and the reference the cached path must reproduce bit for bit.
int run_generate(const GenerateOptions& opt) {
  const auto t_load = std::chrono::steady_clock::now();
  auto loaded = axon::model::LlamaModel::load(opt.model, opt.max_seq);
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
  if (ids.size() > opt.max_seq) {
    fmt::print(stderr, "error: prompt has {} tokens, more than --max-seq {}\n", ids.size(),
               opt.max_seq);
    return 2;
  }
  const auto load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

  axon::model::ForwardOptions forward_options;
  if (opt.dump_layer != axon::model::ForwardOptions::kNoDump) {
    forward_options.dump_layer = opt.dump_layer;
    forward_options.dump_dir = opt.dump_dir;
    std::filesystem::create_directories(forward_options.dump_dir);
  }

  fmt::print("{}", prompt);
  std::fflush(stdout);
  axon::StreamingDecoder decoder(tokenizer);
  const auto eog = tokenizer.eog_tokens();
  const auto t_gen = std::chrono::steady_clock::now();
  int generated = 0;
  const bool dump_only =
      opt.n_tokens <= 0 && opt.dump_layer != axon::model::ForwardOptions::kNoDump;
  if (dump_only) {
    // Nothing to generate, but the dump still needs one pass over the prompt.
    const auto logits = model.forward(ids, forward_options);
    if (!logits) {
      fmt::print(stderr, "\nerror: {}\n", logits.error().message);
      return 2;
    }
  }
  for (; generated < opt.n_tokens; ++generated) {
    if (ids.size() >= opt.max_seq) {
      break;
    }
    // Only the first step writes the dump; later steps would overwrite it with longer rows.
    const auto logits =
        model.forward(ids, generated == 0 ? forward_options : axon::model::ForwardOptions{});
    if (!logits) {
      fmt::print(stderr, "\nerror: {}\n", logits.error().message);
      return 2;
    }
    const std::size_t n_vocab = model.config().n_vocab;
    const float* last = logits->data() + (ids.size() - 1) * n_vocab;
    const auto next =
        static_cast<std::int32_t>(std::ranges::max_element(last, last + n_vocab) - last);
    if (std::ranges::find(eog, next) != eog.end()) {
      break;
    }
    ids.push_back(next);
    fmt::print("{}", decoder.push(next));
    std::fflush(stdout);
  }
  fmt::print("{}\n", decoder.flush());
  const auto gen_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen).count();
  fmt::print("generated {} tokens in {:.2f} s ({:.2f} tok/s), load {:.2f} s, arena {:.1f} MB\n",
             generated, gen_seconds,
             generated > 0 ? static_cast<double>(generated) / gen_seconds : 0.0, load_seconds,
             static_cast<double>(model.arena_high_water()) / (1024.0 * 1024.0));
  if (opt.dump_layer != axon::model::ForwardOptions::kNoDump) {
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
  gen->add_option("--max-seq", generate.max_seq, "prompt plus generated token limit")
      ->default_val(256);
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
