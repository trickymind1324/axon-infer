#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <string_view>
#include <utility>

// fmt-based leveled logging to stderr. Never called from kernels.
namespace axon::log {

enum class Level : std::uint8_t { Trace, Debug, Info, Warn, Error, Off };

void set_level(Level level);
Level level();
bool parse_level(std::string_view text, Level& out);
std::string_view level_name(Level level);

namespace detail {
void write(Level level, std::string_view message);
inline bool enabled(Level lvl) {
  return lvl >= level();
}
}  // namespace detail

template <class... Args> void trace(fmt::format_string<Args...> format, Args&&... args) {
  if (detail::enabled(Level::Trace)) {
    detail::write(Level::Trace, fmt::format(format, std::forward<Args>(args)...));
  }
}

template <class... Args> void debug(fmt::format_string<Args...> format, Args&&... args) {
  if (detail::enabled(Level::Debug)) {
    detail::write(Level::Debug, fmt::format(format, std::forward<Args>(args)...));
  }
}

template <class... Args> void info(fmt::format_string<Args...> format, Args&&... args) {
  if (detail::enabled(Level::Info)) {
    detail::write(Level::Info, fmt::format(format, std::forward<Args>(args)...));
  }
}

template <class... Args> void warn(fmt::format_string<Args...> format, Args&&... args) {
  if (detail::enabled(Level::Warn)) {
    detail::write(Level::Warn, fmt::format(format, std::forward<Args>(args)...));
  }
}

template <class... Args> void error(fmt::format_string<Args...> format, Args&&... args) {
  if (detail::enabled(Level::Error)) {
    detail::write(Level::Error, fmt::format(format, std::forward<Args>(args)...));
  }
}

}  // namespace axon::log
