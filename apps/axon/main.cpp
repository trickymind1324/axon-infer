#include "axon/sampler.h"
#include "axon/tokenizer.h"
#include "axon/version.h"
#include "core/cpu_info.h"
#include "core/gguf.h"
#include "core/log.h"

#include "chat/template.h"
#include "engine/engine.h"
#include "model/llama.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

std::string read_stdin() {
  return {std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
}

// --- tokenize -----------------------------------------------------------------------------

struct TokenizeCli {
  std::string model;
  std::string text;
  bool no_bos = false;
  bool no_special = false;
  bool ids_only = false;
};

int run_tokenize(const TokenizeCli& opt) {
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
  const std::string text = opt.text.empty() ? read_stdin() : opt.text;
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

// --- shared generation pieces -------------------------------------------------------------

struct SamplingCli {
  float temperature = 0.0F;
  int top_k = 40;
  float top_p = 0.95F;
  float min_p = 0.05F;
  float repeat_penalty = 1.1F;
  int repeat_last_n = 64;
  std::uint64_t seed = 0;

  [[nodiscard]] axon::SamplerConfig config() const {
    return {.temperature = temperature,
            .top_k = top_k,
            .top_p = top_p,
            .min_p = min_p,
            .repeat_penalty = repeat_penalty,
            .repeat_last_n = repeat_last_n,
            .seed = seed};
  }
};

// generate defaults to pure greedy so its output matches Hugging Face token for token. chat
// defaults to the usual sampling settings.
void add_sampling_options(CLI::App& cmd, SamplingCli& s, float default_temperature,
                          float default_repeat_penalty) {
  s.temperature = default_temperature;
  s.repeat_penalty = default_repeat_penalty;
  cmd.add_option("--temp", s.temperature, "sampling temperature, 0 is greedy")
      ->default_val(default_temperature);
  cmd.add_option("--top-k", s.top_k, "keep the k most likely tokens, 0 disables")->default_val(40);
  cmd.add_option("--top-p", s.top_p, "nucleus mass, 1 disables")->default_val(0.95F);
  cmd.add_option("--min-p", s.min_p, "drop tokens below this fraction of the best, 0 disables")
      ->default_val(0.05F);
  cmd.add_option("--repeat-penalty", s.repeat_penalty, "penalty for recent tokens, 1 disables")
      ->default_val(default_repeat_penalty);
  cmd.add_option("--repeat-last-n", s.repeat_last_n, "how many recent tokens the penalty sees")
      ->default_val(64);
  cmd.add_option("--seed", s.seed, "random seed, 0 picks one and reports it");
}

std::string stats_line(const axon::engine::Stats& s) {
  return fmt::format("prompt {} tokens ({} reused), first token in {:.2f} s, decode {} tokens "
                     "in {:.2f} s ({:.2f} tok/s), cache {} positions ({:.1f} MB of {:.1f} MB), "
                     "peak rss {:.0f} MB, seed {}",
                     s.prompt_tokens, s.reused_tokens, s.prefill_s, s.generated_tokens, s.decode_s,
                     s.decode_tok_s(), s.cache_positions,
                     static_cast<double>(s.cache_bytes_live) / (1024.0 * 1024.0),
                     static_cast<double>(s.cache_bytes_reserved) / (1024.0 * 1024.0), s.peak_rss_mb,
                     s.seed);
}

// Encodes prompt text for the model. The template may already spell out the BOS token (Llama 3
// does), in which case the tokenizer must not add a second one.
std::vector<std::int32_t> encode_prompt(const axon::Tokenizer& tokenizer, const std::string& text,
                                        bool no_bos) {
  std::optional<bool> add_bos = no_bos ? std::optional<bool>(false) : std::nullopt;
  if (!no_bos && text.starts_with(tokenizer.token_text(tokenizer.bos()))) {
    add_bos = false;
  }
  return tokenizer.encode(text, {.add_bos = add_bos, .parse_special = true});
}

// --- generate -----------------------------------------------------------------------------

struct GenerateCli {
  std::string model;
  std::string prompt;
  int n_tokens = 20;
  std::size_t max_seq = 256;
  std::size_t ctx = 2048;
  int threads = 0;
  bool no_cache = false;
  bool no_bos = false;
  bool stats = false;
  SamplingCli sampling;
  std::size_t dump_layer = axon::model::ForwardOptions::kNoDump;
  std::string dump_dir = "dump-axon";
};

// The cache-free loop: recompute the whole sequence each step, greedy only. Kept as the reference
// and as the only path that can dump activations.
int generate_without_cache(axon::model::LlamaModel& model, std::vector<std::int32_t> ids,
                           const GenerateCli& opt) {
  axon::model::ForwardOptions forward_options;
  if (opt.dump_layer != axon::model::ForwardOptions::kNoDump) {
    forward_options.dump_layer = opt.dump_layer;
    forward_options.dump_dir = opt.dump_dir;
    std::filesystem::create_directories(forward_options.dump_dir);
  }
  const auto& tokenizer = model.tokenizer();
  const std::size_t n_vocab = model.config().n_vocab;
  const auto eog = tokenizer.eog_tokens();
  axon::StreamingDecoder decoder(tokenizer);
  const auto t0 = std::chrono::steady_clock::now();
  int generated = 0;
  bool first = true;
  while (true) {
    const auto logits = model.forward(ids, first ? forward_options : axon::model::ForwardOptions{});
    first = false;
    if (!logits) {
      fmt::print(stderr, "\nerror: {}\n", logits.error().message);
      return 2;
    }
    const float* last = logits->data() + (ids.size() - 1) * n_vocab;
    const auto next =
        static_cast<std::int32_t>(std::ranges::max_element(last, last + n_vocab) - last);
    if (generated >= opt.n_tokens || std::ranges::find(eog, next) != eog.end() ||
        ids.size() >= model.max_seq()) {
      break;
    }
    ids.push_back(next);
    ++generated;
    fmt::print("{}", decoder.push(next));
    std::fflush(stdout);
  }
  fmt::print("{}\n", decoder.flush());
  const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fmt::print("generated {} tokens in {:.2f} s ({:.2f} tok/s) without a cache, arena {:.1f} MB\n",
             generated, seconds, generated > 0 ? generated / seconds : 0.0,
             static_cast<double>(model.arena_high_water()) / (1024.0 * 1024.0));
  if (opt.dump_layer != axon::model::ForwardOptions::kNoDump) {
    fmt::print("layer {} activations written to {}\n", opt.dump_layer, opt.dump_dir);
  }
  return 0;
}

int run_generate(const GenerateCli& opt) {
  const bool use_cache = !opt.no_cache && opt.dump_layer == axon::model::ForwardOptions::kNoDump;
  const std::size_t ctx = use_cache ? opt.ctx : opt.max_seq;
  const auto t_load = std::chrono::steady_clock::now();
  auto loaded = axon::model::LlamaModel::load(
      opt.model, {.max_seq = std::min(opt.max_seq, ctx), .max_ctx = ctx, .n_threads = opt.threads});
  if (!loaded) {
    fmt::print(stderr, "error: {}\n", loaded.error().message);
    return 2;
  }
  auto& model = **loaded;
  const std::string prompt = opt.prompt.empty() ? read_stdin() : opt.prompt;
  const auto ids = encode_prompt(model.tokenizer(), prompt, opt.no_bos);
  const auto load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
  fmt::print("model: {} ({}), loaded in {:.2f} s\n", model.file().model_name(),
             model.config().describe(), load_s);
  fmt::print("prompt: {} tokens\n", ids.size());
  fmt::print("{}", prompt);
  std::fflush(stdout);
  if (!use_cache) {
    if (opt.sampling.temperature > 0.0F) {
      fmt::print(stderr,
                 "\nnote: --no-cache and --dump-layer are greedy, sampling flags are ignored\n");
    }
    return generate_without_cache(model, ids, opt);
  }

  axon::engine::Engine engine(model);
  axon::engine::GenerateOptions options;
  options.max_tokens = opt.n_tokens;
  options.sampler = opt.sampling.config();
  const auto out = engine.generate(ids, options, [](std::string_view piece) {
    fmt::print("{}", piece);
    std::fflush(stdout);
  });
  fmt::print("\n");
  if (!out) {
    fmt::print(stderr, "error: {}\n", out.error().message);
    return 2;
  }
  const auto& s = engine.stats();
  fmt::print("{}\n", stats_line(s));
  if (s.context_full) {
    fmt::print("stopped: the context of {} is full, raise --ctx for longer output\n",
               model.max_ctx());
  }
  return 0;
}

// --- chat ---------------------------------------------------------------------------------

struct ChatCli {
  std::string model;
  std::string system;
  std::size_t ctx = 2048;
  std::size_t max_seq = 256;
  int max_tokens = 256;
  int threads = 0;
  bool stats = false;
  SamplingCli sampling;
};

int run_chat(const ChatCli& opt) {
  const auto t_load = std::chrono::steady_clock::now();
  auto loaded = axon::model::LlamaModel::load(
      opt.model,
      {.max_seq = std::min(opt.max_seq, opt.ctx), .max_ctx = opt.ctx, .n_threads = opt.threads});
  if (!loaded) {
    fmt::print(stderr, "error: {}\n", loaded.error().message);
    return 2;
  }
  auto& model = **loaded;
  const auto& tokenizer = model.tokenizer();
  const auto tmpl = axon::chat::ChatTemplate::from_tokenizer(tokenizer);
  axon::engine::Engine engine(model);
  axon::engine::GenerateOptions options;
  options.max_tokens = opt.max_tokens;
  options.sampler = opt.sampling.config();
  const auto load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

  fmt::print("model: {} ({}), loaded in {:.2f} s, {} threads\n", model.file().model_name(),
             model.config().describe(), load_s, model.n_threads());
  fmt::print("chat template: {}, context {} tokens, temperature {}\n", tmpl.source(),
             model.max_ctx(), opt.sampling.temperature);
  fmt::print("type a message and press enter. /clear starts over, /stats shows the last turn, "
             "/exit quits.\n\n");

  std::vector<axon::chat::Message> messages;
  const auto reset = [&] {
    messages.clear();
    if (!opt.system.empty()) {
      messages.push_back({.role = "system", .content = opt.system});
    }
    engine.clear();
  };
  reset();
  bool have_stats = false;

  std::string line;
  while (true) {
    fmt::print("> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) {
      fmt::print("\n");
      break;
    }
    while (!line.empty() && (line.back() == ' ' || line.back() == '\r')) {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    if (line == "/exit" || line == "/quit") {
      break;
    }
    if (line == "/clear") {
      reset();
      fmt::print("(conversation cleared)\n");
      continue;
    }
    if (line == "/stats") {
      fmt::print("{}\n", have_stats ? stats_line(engine.stats()) : "no turn yet");
      continue;
    }
    messages.push_back({.role = "user", .content = line});
    const std::string prompt = tmpl.render(messages, true);
    const auto ids = encode_prompt(tokenizer, prompt, false);
    std::string reply;
    const auto out = engine.generate(ids, options, [&](std::string_view piece) {
      reply.append(piece);
      fmt::print("{}", piece);
      std::fflush(stdout);
    });
    fmt::print("\n");
    if (!out) {
      fmt::print("(cannot continue: {}. Use /clear to start over or raise --ctx.)\n",
                 out.error().message);
      messages.pop_back();
      continue;
    }
    have_stats = true;
    messages.push_back({.role = "assistant", .content = reply});
    if (engine.stats().context_full) {
      fmt::print("(the context of {} tokens is full. /clear starts over.)\n", model.max_ctx());
    }
    if (opt.stats) {
      fmt::print("[{}]\n", stats_line(engine.stats()));
    }
  }
  return 0;
}

// --- main ---------------------------------------------------------------------------------

int run(int argc, char** argv) {
  CLI::App app{"axon-infer: LLM inference engine in C++20"};
  app.set_version_flag("--version", [] {
    print_version();
    return std::string{};
  });
  std::string log_level = "info";
  app.add_option("--log-level", log_level, "trace|debug|info|warn|error|off")->default_str("info");

  TokenizeCli tokenize;
  auto* tok =
      app.add_subcommand("tokenize", "encode text with a model's tokenizer and print the tokens");
  tok->add_option("--model,-m", tokenize.model, "path to a .gguf file")->required();
  tok->add_option("text", tokenize.text, "text to encode (reads stdin when omitted)");
  tok->add_flag("--no-bos", tokenize.no_bos, "do not prepend the BOS token");
  tok->add_flag("--no-special", tokenize.no_special,
                "treat special tokens in the text as plain text");
  tok->add_flag("--ids-only", tokenize.ids_only, "print only the ids on one line");

  GenerateCli generate;
  auto* gen = app.add_subcommand("generate", "complete a prompt");
  gen->add_option("--model,-m", generate.model, "path to a .gguf file")->required();
  gen->add_option("--prompt,-p", generate.prompt, "prompt text (reads stdin when omitted)");
  gen->add_option("--n-tokens,-n", generate.n_tokens, "tokens to generate")->default_val(20);
  gen->add_option("--max-seq", generate.max_seq, "tokens per forward call (prompt chunk size)")
      ->default_val(256);
  gen->add_option("--ctx", generate.ctx, "key/value cache length (prompt plus generated tokens)")
      ->default_val(2048);
  gen->add_option("--threads,-t", generate.threads,
                  "threads for the forward pass, 0 for one per core")
      ->default_val(0);
  gen->add_flag("--no-cache", generate.no_cache,
                "recompute the whole sequence every step instead of using the cache (greedy)");
  gen->add_flag("--no-bos", generate.no_bos, "do not prepend the BOS token");
  gen->add_option(
      "--dump-layer", generate.dump_layer,
      "write this layer's activations from the first forward pass (implies --no-cache)");
  gen->add_option("--dump-dir", generate.dump_dir, "directory for --dump-layer")
      ->default_str("dump-axon");
  add_sampling_options(*gen, generate.sampling, 0.0F, 1.0F);

  ChatCli chat;
  auto* ch = app.add_subcommand("chat", "multi-turn conversation using the model's chat template");
  ch->add_option("--model,-m", chat.model, "path to a .gguf file")->required();
  ch->add_option("--system", chat.system, "system prompt for the conversation");
  ch->add_option("--ctx", chat.ctx, "key/value cache length")->default_val(2048);
  ch->add_option("--max-seq", chat.max_seq, "tokens per forward call")->default_val(256);
  ch->add_option("--max-tokens", chat.max_tokens, "longest reply in tokens")->default_val(256);
  ch->add_option("--threads,-t", chat.threads, "threads for the forward pass, 0 for one per core")
      ->default_val(0);
  ch->add_flag("--stats", chat.stats, "print timing and memory after every reply");
  add_sampling_options(*ch, chat.sampling, 0.8F, 1.1F);

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
  if (ch->parsed()) {
    return run_chat(chat);
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
