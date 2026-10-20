// axon-server: one model behind the OpenAI chat completions API. Requests are served in
// arrival order by a single worker, with a bounded queue, streaming over server-sent events,
// and an optional bearer token.

#include "axon/version.h"
#include "core/cpu_info.h"
#include "core/log.h"
#include "core/threadpool.h"

#include "kernels/kernels.h"
#include "model/llama.h"
#include "server/http.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) {
  g_stop.store(true);
}

std::string compiler_string() {
#if defined(__clang__)
  return fmt::format("clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  return fmt::format("gcc {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

struct Cli {
  std::string model;
  std::string host = "127.0.0.1";
  int port = 8080;
  int threads = 0;
  std::size_t ctx = 4096;
  std::size_t max_seq = 256;
  std::string cache = "f32";
  bool force_scalar = false;
  int max_tokens = 512;
  std::size_t max_queue = 16;
  std::size_t max_batch = 4;
  std::string api_key;
  std::string served_name;
  std::string log_level = "info";
};

int run(int argc, char** argv) {
  CLI::App app{"axon-server: one model behind the OpenAI chat completions API"};
  app.set_version_flag("--version", [] {
    fmt::print("axon-infer {} (commit {})\n", axon::version(), axon::git_commit());
    fmt::print("compiler: {}\n", compiler_string());
    fmt::print("cpu: {}\n", axon::cpu_info().summary());
    return std::string{};
  });
  Cli opt;
  app.add_option("--model,-m", opt.model, "path to a .gguf file")->required();
  app.add_option("--host", opt.host, "bind address, 0.0.0.0 for containers")
      ->default_str("127.0.0.1");
  app.add_option("--port", opt.port, "TCP port")->default_val(8080);
  app.add_option("--threads,-t", opt.threads, "threads for the forward pass, 0 for one per core")
      ->default_val(0);
  app.add_option("--ctx", opt.ctx, "key/value cache length, the limit on prompt plus completion")
      ->default_val(4096);
  app.add_option("--max-seq", opt.max_seq, "tokens per forward call (prompt chunk size)")
      ->default_val(256);
  app.add_option("--cache", opt.cache, "key/value cache storage, f32 or f16")->default_str("f32");
  app.add_flag("--force-scalar", opt.force_scalar, "use the scalar reference kernels");
  app.add_option("--max-tokens", opt.max_tokens, "completion length when the request sets none")
      ->default_val(512);
  app.add_option("--max-queue", opt.max_queue,
                 "requests waiting for the worker before new ones are refused with 503")
      ->default_val(16);
  app.add_option("--max-batch", opt.max_batch,
                 "sequences decoded together, each with its own cache of --ctx positions")
      ->default_val(4);
  app.add_option("--api-key", opt.api_key,
                 "when set, /v1 requests need 'Authorization: Bearer <key>'");
  app.add_option("--served-name", opt.served_name,
                 "the model id reported by /v1/models (default: the file's stem)");
  app.add_option("--log-level", opt.log_level, "trace|debug|info|warn|error|off")
      ->default_str("info");
  CLI11_PARSE(app, argc, argv);

  axon::log::Level level{};
  if (!axon::log::parse_level(opt.log_level, level)) {
    fmt::print(stderr, "error: --log-level: expected one of trace|debug|info|warn|error|off\n");
    return 2;
  }
  axon::log::set_level(level);
  if (opt.cache != "f32" && opt.cache != "f16") {
    fmt::print(stderr, "error: --cache: expected f32 or f16, found '{}'\n", opt.cache);
    return 2;
  }
  if (opt.port < 0 || opt.port > 65535) {
    fmt::print(stderr, "error: --port: expected 0 to 65535\n");
    return 2;
  }

  const auto t_load = std::chrono::steady_clock::now();
  auto loaded = axon::model::LlamaModel::load(
      opt.model,
      {.max_seq = std::min(opt.max_seq, opt.ctx),
       .max_ctx = opt.ctx,
       .n_threads = opt.threads,
       .kernels =
           opt.force_scalar ? axon::kernels::KernelSet::Scalar : axon::kernels::KernelSet::Auto,
       .max_batch = std::max<std::size_t>(1, opt.max_batch),
       .cache = opt.cache == "f16" ? axon::model::CacheType::F16 : axon::model::CacheType::F32});
  if (!loaded) {
    fmt::print(stderr, "error: {}\n", loaded.error().message);
    return 2;
  }
  auto& model = **loaded;
  const auto load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

  axon::server::ServerOptions options{
      .host = opt.host,
      .port = opt.port,
      .max_queue = opt.max_queue,
      .max_batch = opt.max_batch,
      .api_key = opt.api_key,
      .served_name = opt.served_name.empty() ? std::filesystem::path(opt.model).stem().string()
                                             : opt.served_name,
      .default_max_tokens = opt.max_tokens};
  axon::server::HttpServer server(model, options);
  if (!server.start()) {
    fmt::print(stderr, "error: cannot listen on {}:{}\n", opt.host, opt.port);
    return 2;
  }
  fmt::print("model: {} ({}), loaded in {:.2f} s, {} threads, {} kernels, context {}, {} cache\n",
             model.file().model_name(), model.config().describe(), load_s, model.n_threads(),
             axon::kernels::name(model.table()), model.max_ctx(), opt.cache);
  fmt::print("serving '{}' at http://{}:{}/v1 ({}), up to {} sequences at once, queue of {}\n",
             server.served_name(), opt.host, server.port(),
             opt.api_key.empty() ? "no authentication" : "bearer token required", opt.max_batch,
             opt.max_queue);
  std::fflush(stdout);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  fmt::print("\nstopping after {} requests\n", server.requests());
  server.stop();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    fmt::print(stderr, "error: {}\n", e.what());
    return 2;
  }
}
