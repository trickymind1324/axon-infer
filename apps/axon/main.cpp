#include "axon/version.h"
#include "core/cpu_info.h"
#include "core/log.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <exception>
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

int run(int argc, char** argv) {
  CLI::App app{"axon-infer: LLM inference engine in C++20"};
  app.set_version_flag("--version", [] {
    print_version();
    return std::string{};
  });

  std::string log_level = "info";
  app.add_option("--log-level", log_level, "trace|debug|info|warn|error|off")->default_str("info");

  app.require_subcommand(0);
  CLI11_PARSE(app, argc, argv);

  axon::log::Level level{};
  if (!axon::log::parse_level(log_level, level)) {
    fmt::print(stderr,
               "error: --log-level: expected one of trace|debug|info|warn|error|off, found '{}'\n",
               log_level);
    return 2;
  }
  axon::log::set_level(level);

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
