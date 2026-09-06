#include "core/check.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("AXON_CHECK passes silently on true", "[core][check]") {
  int calls = 0;
  AXON_CHECK(++calls == 1);
  AXON_CHECK(calls == 1, "calls={}", calls);
  AXON_DCHECK(calls == 1);
  AXON_DCHECK(calls == 1, "calls={}", calls);
  CHECK(calls == 1);
}

TEST_CASE("check messages are formatted lazily", "[core][check]") {
  CHECK(axon::detail::check_message().empty());
  CHECK(axon::detail::check_message("{} of {}", 3, 4) == "3 of 4");
}
