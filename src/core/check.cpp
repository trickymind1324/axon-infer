#include "core/check.h"

#include "core/log.h"

#include <cstdio>
#include <cstdlib>

namespace axon::detail {

void check_failed(const char* file, int line, const char* expr, const std::string& message) {
  if (message.empty()) {
    log::error("check failed: {} ({}:{})", expr, file, line);
  } else {
    log::error("check failed: {} ({}:{}): {}", expr, file, line, message);
  }
  std::fflush(stderr);
  std::abort();
}

}  // namespace axon::detail
