#pragma once

// Builds small metadata-only GGUF images in memory so tests can probe config errors without
// committing a fixture per case.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace axon::test {

class GGUFBuilder {
public:
  GGUFBuilder& u32(std::string_view key, std::uint32_t value) {
    put_string(key);
    put_u32(4);  // UInt32
    put_u32(value);
    ++n_kv_;
    return *this;
  }
  GGUFBuilder& f32(std::string_view key, float value) {
    put_string(key);
    put_u32(6);  // Float32
    std::uint32_t bits = 0;
    static_assert(sizeof bits == sizeof value);
    __builtin_memcpy(&bits, &value, sizeof bits);
    put_u32(bits);
    ++n_kv_;
    return *this;
  }
  GGUFBuilder& string(std::string_view key, std::string_view value) {
    put_string(key);
    put_u32(8);  // String
    put_string(value);
    ++n_kv_;
    return *this;
  }

  // Header + entries, no tensors. Padded to the default alignment.
  std::vector<std::byte> build() const {
    std::vector<std::byte> out;
    const auto append_u32 = [&](std::uint32_t v) {
      for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
      }
    };
    const auto append_u64 = [&](std::uint64_t v) {
      for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
      }
    };
    append_u32(0x46554747U);
    append_u32(3);
    append_u64(0);
    append_u64(n_kv_);
    out.insert(out.end(), body_.begin(), body_.end());
    while (out.size() % 32 != 0) {
      out.push_back(std::byte{0});
    }
    return out;
  }

private:
  void put_u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      body_.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
    }
  }
  void put_u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      body_.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
    }
  }
  void put_string(std::string_view s) {
    put_u64(s.size());
    for (const char c : s) {
      body_.push_back(static_cast<std::byte>(c));
    }
  }

  std::vector<std::byte> body_;
  std::uint64_t n_kv_ = 0;
};

}  // namespace axon::test
