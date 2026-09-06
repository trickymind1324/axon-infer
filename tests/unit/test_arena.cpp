#include "core/arena.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

#if !defined(AXON_SANITIZE_ENABLED)
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#endif

namespace {

bool is_aligned(const void* pointer, std::size_t align) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & (align - 1)) == 0;
}

}  // namespace

TEST_CASE("Arena returns aligned pointers inside its buffer", "[core][arena]") {
  axon::Arena arena(1 << 20);
  const auto* base = static_cast<const unsigned char*>(arena.base());
  for (const std::size_t align : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL, 256UL, 4096UL}) {
    arena.alloc(1, 1);  // misalign the cursor first
    void* memory = arena.alloc(17, align);
    CHECK(is_aligned(memory, align));
    const auto* bytes = static_cast<const unsigned char*>(memory);
    CHECK(bytes >= base);
    CHECK(bytes + 17 <= base + arena.capacity());
  }
}

TEST_CASE("Arena allocations do not overlap", "[core][arena]") {
  axon::Arena arena(4096);
  std::vector<unsigned char*> blocks;
  for (std::size_t i = 0; i < 16; ++i) {
    auto* block = static_cast<unsigned char*>(arena.alloc(100, 16));
    std::memset(block, static_cast<int>(i + 1), 100);
    blocks.push_back(block);
  }
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    for (std::size_t j = 0; j < 100; ++j) {
      REQUIRE(blocks[i][j] == static_cast<unsigned char>(i + 1));
    }
  }
}

TEST_CASE("Arena reset reuses memory and tracks high water", "[core][arena]") {
  axon::Arena arena(1024);
  CHECK(arena.used() == 0);
  CHECK(arena.high_water() == 0);

  void* first = arena.alloc(300);
  CHECK(arena.used() == 300);
  CHECK(arena.high_water() == 300);

  arena.reset();
  CHECK(arena.used() == 0);
  CHECK(arena.high_water() == 300);

  void* again = arena.alloc(100);
  CHECK(again == first);
  CHECK(arena.high_water() == 300);

  arena.alloc(400);
  CHECK(arena.used() == 128 + 400);  // 100 rounded up to the 64-byte default alignment
  CHECK(arena.high_water() == arena.used());
}

TEST_CASE("Arena alloc_array gives typed spans", "[core][arena]") {
  axon::Arena arena(4096);
  auto floats = arena.alloc_array<float>(100);
  CHECK(floats.size() == 100);
  CHECK(is_aligned(floats.data(), axon::Arena::kDefaultAlign));
  auto bytes = arena.alloc_array<std::uint8_t>(3, 1);
  CHECK(bytes.size() == 3);
  CHECK(arena.used() == 400 + 3);
}

TEST_CASE("Arena move transfers ownership", "[core][arena]") {
  axon::Arena source(2048);
  void* memory = source.alloc(128);
  std::memset(memory, 0x5A, 128);

  axon::Arena target(std::move(source));
  CHECK(target.capacity() == 2048);
  CHECK(target.used() == 128);
  CHECK(static_cast<const unsigned char*>(target.base())[0] == 0x5A);
  CHECK(source.capacity() == 0);  // NOLINT(bugprone-use-after-move): moved-from state is defined
  CHECK(source.base() == nullptr);

  axon::Arena assigned(16);
  assigned = std::move(target);
  CHECK(assigned.capacity() == 2048);
  CHECK(target.capacity() == 0);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("Arena with zero capacity is valid and empty", "[core][arena]") {
  const axon::Arena arena(0);
  CHECK(arena.capacity() == 0);
  CHECK(arena.base() == nullptr);
}

#if !defined(AXON_SANITIZE_ENABLED)
TEST_CASE("Arena overflow aborts", "[core][arena]") {
  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    axon::Arena arena(256);
    arena.alloc(200);
    arena.alloc(200);  // overflow: fatal check
    ::_exit(0);        // not reached
  }
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status));
  CHECK(WTERMSIG(status) == SIGABRT);
}
#endif
