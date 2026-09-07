#include "axon/tokenizer.h"
#include "axon/version.h"
#include "core/cpu_info.h"
#include "core/gguf.h"
#include "core/log.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <exception>
#include <iostream>
#include <iterator>
#include <string>

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
