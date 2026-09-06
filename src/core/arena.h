#pragma once

#include <cstddef>
#include <span>

namespace axon {

// Bump allocator for activation memory. Sized once (from known model shapes), reset per
// forward pass. Overflow is a programmer error and aborts via AXON_CHECK; the arena is
// never asked for memory whose size was not computed at load time.
//
// Backed by anonymous mmap: page aligned, lazily committed, released page-wise.
class Arena {
public:
  static constexpr std::size_t kDefaultAlign = 64;

  explicit Arena(std::size_t capacity_bytes);
  ~Arena();

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&& other) noexcept;
  Arena& operator=(Arena&& other) noexcept;

  // Returns a pointer aligned to `align` (a power of two). Never null.
  void* alloc(std::size_t bytes, std::size_t align = kDefaultAlign);

  template <class T>
  std::span<T> alloc_array(std::size_t count, std::size_t align = alignof(T) > kDefaultAlign
                                                                      ? alignof(T)
                                                                      : kDefaultAlign) {
    void* memory = alloc(count * sizeof(T), align);
    return {static_cast<T*>(memory), count};
  }

  // Discards all allocations. Debug builds poison the previously used region.
  void reset();

  [[nodiscard]] std::size_t used() const noexcept { return used_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t high_water() const noexcept { return high_water_; }
  [[nodiscard]] const void* base() const noexcept { return base_; }

private:
  void release() noexcept;

  unsigned char* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t mapped_bytes_ = 0;
  std::size_t used_ = 0;
  std::size_t high_water_ = 0;
};

}  // namespace axon
