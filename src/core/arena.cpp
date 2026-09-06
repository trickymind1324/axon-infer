#include "core/arena.h"

#include "core/check.h"
#include "core/os.h"

#include <cstring>

namespace axon {

namespace {

constexpr bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

Arena::Arena(std::size_t capacity_bytes) : capacity_(capacity_bytes) {
  if (capacity_bytes == 0) {
    return;
  }
  const std::size_t page = os::page_size();
  mapped_bytes_ = (capacity_bytes + page - 1) / page * page;
  base_ = static_cast<unsigned char*>(os::map_anonymous(mapped_bytes_));
  AXON_CHECK(base_ != nullptr, "arena: failed to map {} bytes", mapped_bytes_);
}

Arena::~Arena() {
  release();
}

Arena::Arena(Arena&& other) noexcept
    : base_(other.base_), capacity_(other.capacity_), mapped_bytes_(other.mapped_bytes_),
      used_(other.used_), high_water_(other.high_water_) {
  other.base_ = nullptr;
  other.capacity_ = 0;
  other.mapped_bytes_ = 0;
  other.used_ = 0;
  other.high_water_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
  if (this != &other) {
    release();
    base_ = other.base_;
    capacity_ = other.capacity_;
    mapped_bytes_ = other.mapped_bytes_;
    used_ = other.used_;
    high_water_ = other.high_water_;
    other.base_ = nullptr;
    other.capacity_ = 0;
    other.mapped_bytes_ = 0;
    other.used_ = 0;
    other.high_water_ = 0;
  }
  return *this;
}

void Arena::release() noexcept {
  if (base_ != nullptr) {
    os::unmap(base_, mapped_bytes_);
    base_ = nullptr;
  }
  capacity_ = 0;
  mapped_bytes_ = 0;
  used_ = 0;
  high_water_ = 0;
}

void* Arena::alloc(std::size_t bytes, std::size_t align) {
  AXON_CHECK(is_power_of_two(align), "arena: alignment {} is not a power of two", align);
  const std::size_t start = (used_ + align - 1) & ~(align - 1);
  AXON_CHECK(start >= used_ && start + bytes >= start && start + bytes <= capacity_,
             "arena: out of memory: requested {} bytes (align {}), used {} of {}", bytes, align,
             used_, capacity_);
  used_ = start + bytes;
  if (used_ > high_water_) {
    high_water_ = used_;
  }
  return base_ + start;
}

void Arena::reset() {
#ifndef NDEBUG
  if (base_ != nullptr && used_ > 0) {
    std::memset(base_, 0xAA, used_);
  }
#endif
  used_ = 0;
}

}  // namespace axon
