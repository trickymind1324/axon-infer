#include "core/expected.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

namespace {

axon::Expected<int> parse_positive(int value) {
  if (value <= 0) {
    return axon::make_error(axon::ErrorCode::InvalidArgument,
                            "expected a positive integer, found " + std::to_string(value));
  }
  return value;
}

axon::Expected<void> check_even(int value) {
  if (value % 2 != 0) {
    return axon::make_error(axon::ErrorCode::InvalidArgument, "expected even");
  }
  return {};
}

}  // namespace

TEST_CASE("Expected holds a value", "[core][expected]") {
  const auto result = parse_positive(7);
  REQUIRE(result.has_value());
  REQUIRE(static_cast<bool>(result));
  CHECK(result.value() == 7);
  CHECK(*result == 7);
  CHECK(result.value_or(-1) == 7);
}

TEST_CASE("Expected holds an error", "[core][expected]") {
  const auto result = parse_positive(-3);
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == axon::ErrorCode::InvalidArgument);
  CHECK(result.error().message == "expected a positive integer, found -3");
  CHECK(result.value_or(-1) == -1);
  CHECK(std::string(axon::error_code_name(result.error().code)) == "invalid_argument");
}

TEST_CASE("Expected<void>", "[core][expected]") {
  CHECK(check_even(4).has_value());
  const auto odd = check_even(5);
  REQUIRE_FALSE(odd);
  CHECK(odd.error().message == "expected even");
}

TEST_CASE("Expected with move-only payload", "[core][expected]") {
  axon::Expected<std::unique_ptr<int>> result = std::make_unique<int>(42);
  REQUIRE(result);
  CHECK(**result == 42);
  auto moved = std::move(result).value();
  CHECK(*moved == 42);

  axon::Expected<std::unique_ptr<int>> other = axon::make_error(axon::ErrorCode::Io, "disk");
  axon::Expected<std::unique_ptr<int>> assigned = std::move(other);
  REQUIRE_FALSE(assigned);
  CHECK(assigned.error().code == axon::ErrorCode::Io);
}

TEST_CASE("Expected copy and assignment across states", "[core][expected]") {
  axon::Expected<std::string> value = std::string("hello");
  axon::Expected<std::string> error = axon::make_error(axon::ErrorCode::Parse, "bad");

  axon::Expected<std::string> copy = value;
  CHECK(copy.value() == "hello");

  copy = error;
  REQUIRE_FALSE(copy);
  CHECK(copy.error().message == "bad");

  copy = value;
  REQUIRE(copy);
  CHECK(copy->size() == 5);

  value = axon::Expected<std::string>(axon::make_error(axon::ErrorCode::Internal, "x"));
  REQUIRE_FALSE(value);
  CHECK(value.error().code == axon::ErrorCode::Internal);
}
