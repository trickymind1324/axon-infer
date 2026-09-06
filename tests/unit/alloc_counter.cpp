#include "alloc_counter.h"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

std::atomic<std::size_t> g_allocations{0};

void* counted_alloc(std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (size == 0) {
    size = 1;
  }
  void* memory = std::malloc(size);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  return memory;
}

}  // namespace

namespace axon::test {

std::size_t allocation_count() {
  return g_allocations.load(std::memory_order_relaxed);
}

}  // namespace axon::test

void* operator new(std::size_t size) {
  return counted_alloc(size);
}
void* operator new[](std::size_t size) {
  return counted_alloc(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size == 0 ? 1 : size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size == 0 ? 1 : size);
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete[](void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
  std::free(memory);
}
