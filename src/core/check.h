#pragma once

#include <fmt/format.h>

#include <cstdlib>

namespace axon::detail {

[[noreturn]] void check_failed(const char* file, int line, const char* expr,
                               const std::string& message);

}  // namespace axon::detail

// Fatal invariant for programmer errors (never for user input). Prints and aborts.
// AXON_CHECK is always on; AXON_DCHECK compiles out in release builds.
#define AXON_CHECK(cond, ...)                                                                      \
  do {                                                                                             \
    if (!(cond)) [[unlikely]] {                                                                    \
      ::axon::detail::check_failed(__FILE__, __LINE__, #cond,                                      \
                                   ::axon::detail::check_message(__VA_ARGS__));                    \
    }                                                                                              \
  } while (false)

#ifdef NDEBUG
#define AXON_DCHECK(cond, ...)                                                                     \
  do {                                                                                             \
    if (false) {                                                                                   \
      static_cast<void>(cond);                                                                     \
    }                                                                                              \
  } while (false)
#else
#define AXON_DCHECK(cond, ...) AXON_CHECK(cond, __VA_ARGS__)
#endif

namespace axon::detail {

inline std::string check_message() {
  return {};
}

template <class... Args>
std::string check_message(fmt::format_string<Args...> format, Args&&... args) {
  return fmt::format(format, std::forward<Args>(args)...);
}

}  // namespace axon::detail
