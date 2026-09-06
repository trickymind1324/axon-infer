#include "core/log.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("log level names round-trip through parse_level", "[core][log]") {
  for (const axon::log::Level lvl :
       {axon::log::Level::Trace, axon::log::Level::Debug, axon::log::Level::Info,
        axon::log::Level::Warn, axon::log::Level::Error, axon::log::Level::Off}) {
    axon::log::Level parsed{};
    REQUIRE(axon::log::parse_level(axon::log::level_name(lvl), parsed));
    CHECK(parsed == lvl);
  }
  axon::log::Level unused{};
  CHECK_FALSE(axon::log::parse_level("verbose", unused));
  CHECK_FALSE(axon::log::parse_level("", unused));
}

TEST_CASE("log level is settable and gates output", "[core][log]") {
  const axon::log::Level previous = axon::log::level();
  axon::log::set_level(axon::log::Level::Off);
  CHECK(axon::log::level() == axon::log::Level::Off);
  // Nothing is written at Off; these calls must be safe to make.
  axon::log::trace("hidden {}", 1);
  axon::log::debug("hidden {}", 2);
  axon::log::info("hidden {}", 3);
  axon::log::warn("hidden {}", 4);
  axon::log::error("hidden {}", 5);

  axon::log::set_level(axon::log::Level::Warn);
  CHECK(axon::log::level() == axon::log::Level::Warn);
  axon::log::warn("axon log test: warn line is expected here");
  axon::log::set_level(previous);
}
