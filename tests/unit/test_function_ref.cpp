#include "core/function_ref.h"

#include "alloc_counter.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>

namespace {

int add_one(int value) {
  return value + 1;
}

int call_twice(axon::FunctionRef<int(int)> fn, int value) {
  return fn(fn(value));
}

}  // namespace

TEST_CASE("FunctionRef calls free functions and lambdas", "[core][function_ref]") {
  CHECK(call_twice(add_one, 1) == 3);
  CHECK(call_twice([](int v) { return v * 2; }, 3) == 12);

  int captured = 10;
  auto capturing = [&captured](int v) { return v + captured; };
  CHECK(call_twice(capturing, 1) == 21);
}

TEST_CASE("FunctionRef forwards mutable state through captures", "[core][function_ref]") {
  std::array<std::size_t, 4> hits{};
  auto counter = [&hits](std::size_t index) { ++hits[index]; };
  axon::FunctionRef<void(std::size_t)> ref = counter;
  ref(0);
  ref(0);
  ref(3);
  CHECK(hits == std::array<std::size_t, 4>{2, 0, 0, 1});
}

TEST_CASE("FunctionRef is trivially copyable and small", "[core][function_ref]") {
  STATIC_CHECK(sizeof(axon::FunctionRef<void()>) == 2 * sizeof(void*));
  STATIC_CHECK(std::is_trivially_copyable_v<axon::FunctionRef<void()>>);
}

TEST_CASE("FunctionRef never allocates", "[core][function_ref]") {
  std::array<int, 64> big{};
  auto heavy = [big](int v) mutable { return v + big[0]; };  // large capture, still no heap
  const std::size_t before = axon::test::allocation_count();
  axon::FunctionRef<int(int)> ref = heavy;
  const int result = ref(1) + ref(2);
  const std::size_t after = axon::test::allocation_count();
  CHECK(result == 3);
  CHECK(after == before);
}
