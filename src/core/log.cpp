#include "core/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace axon::log {

namespace {

std::atomic<Level> g_level{Level::Info};
std::mutex g_write_mutex;

}  // namespace

void set_level(Level level) {
  g_level.store(level, std::memory_order_relaxed);
}

Level level() {
  return g_level.load(std::memory_order_relaxed);
}

std::string_view level_name(Level level) {
  switch (level) {
  case Level::Trace:
    return "trace";
  case Level::Debug:
    return "debug";
  case Level::Info:
    return "info";
  case Level::Warn:
    return "warn";
  case Level::Error:
    return "error";
  case Level::Off:
    return "off";
  }
  return "unknown";
}

bool parse_level(std::string_view text, Level& out) {
  for (Level lvl :
       {Level::Trace, Level::Debug, Level::Info, Level::Warn, Level::Error, Level::Off}) {
    if (text == level_name(lvl)) {
      out = lvl;
      return true;
    }
  }
  return false;
}

namespace detail {

void write(Level level, std::string_view message) {
  const std::scoped_lock lock(g_write_mutex);
  fmt::print(stderr, "[axon][{}] {}\n", level_name(level), message);
}

}  // namespace detail

}  // namespace axon::log
