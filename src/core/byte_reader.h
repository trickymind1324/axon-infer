#pragma once

#include "core/expected.h"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace axon {

// Little-endian, bounds-checked cursor over a byte span. Never throws, never reads past the
// end; every failure names what was being read and where.
class ByteReader {
public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }
  [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
  [[nodiscard]] const std::byte* current() const noexcept { return bytes_.data() + offset_; }

  Expected<std::uint8_t> u8(std::string_view what) { return scalar<std::uint8_t>(what); }
  Expected<std::int8_t> i8(std::string_view what) { return scalar<std::int8_t>(what); }
  Expected<std::uint16_t> u16(std::string_view what) { return scalar<std::uint16_t>(what); }
  Expected<std::int16_t> i16(std::string_view what) { return scalar<std::int16_t>(what); }
  Expected<std::uint32_t> u32(std::string_view what) { return scalar<std::uint32_t>(what); }
  Expected<std::int32_t> i32(std::string_view what) { return scalar<std::int32_t>(what); }
  Expected<std::uint64_t> u64(std::string_view what) { return scalar<std::uint64_t>(what); }
  Expected<std::int64_t> i64(std::string_view what) { return scalar<std::int64_t>(what); }
  Expected<float> f32(std::string_view what) { return scalar<float>(what); }
  Expected<double> f64(std::string_view what) { return scalar<double>(what); }

  // u64 length prefix followed by bytes. The view points into the underlying span.
  Expected<std::string_view> string(std::string_view what, std::size_t max_bytes) {
    const std::size_t start = offset_;
    auto len = u64(what);
    if (!len) {
      return len.error();
    }
    if (*len > max_bytes) {
      return fail(what, start,
                  fmt::format("string length {} exceeds the limit of {}", *len, max_bytes));
    }
    if (*len > remaining()) {
      return fail(what, start,
                  fmt::format("string length {} exceeds the {} bytes left", *len, remaining()));
    }
    const auto* chars = reinterpret_cast<const char*>(current());
    offset_ += static_cast<std::size_t>(*len);
    return std::string_view(chars, static_cast<std::size_t>(*len));
  }

  Expected<void> skip(std::size_t count, std::string_view what) {
    if (count > remaining()) {
      return fail(what, offset_, fmt::format("needs {} bytes, {} left", count, remaining()));
    }
    offset_ += count;
    return {};
  }

  Expected<void> seek(std::size_t absolute, std::string_view what) {
    if (absolute > bytes_.size()) {
      return fail(what, offset_,
                  fmt::format("offset {} is beyond the {} byte file", absolute, bytes_.size()));
    }
    offset_ = absolute;
    return {};
  }

private:
  template <class T> Expected<T> scalar(std::string_view what) {
    if (sizeof(T) > remaining()) {
      return fail(what, offset_, fmt::format("needs {} bytes, {} left", sizeof(T), remaining()));
    }
    T value;
    std::memcpy(&value, current(), sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  static Error fail(std::string_view what, std::size_t at, const std::string& detail) {
    return make_error(ErrorCode::Parse, fmt::format("{} at offset {:#x}: {}", what, at, detail));
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0;
};

}  // namespace axon
