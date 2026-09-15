#pragma once

// Reader for the .axt tensors written by tests/reference/gen_kernel_fixtures.py.

#include "core/dtype.h"

#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace axon::test {

enum class AxtType : std::uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3 };

struct FixtureTensor {
  std::vector<std::uint64_t> dims;
  AxtType type = AxtType::F32;
  std::vector<std::byte> data;

  std::size_t numel() const {
    std::size_t n = 1;
    for (const auto d : dims) {
      n *= static_cast<std::size_t>(d);
    }
    return n;
  }
  std::size_t dim(std::size_t i) const { return static_cast<std::size_t>(dims.at(i)); }

  // Element values as float, converting F16 and BF16 exactly.
  std::vector<float> as_f32() const {
    std::vector<float> out(numel());
    if (type == AxtType::F32) {
      std::memcpy(out.data(), data.data(), out.size() * sizeof(float));
      return out;
    }
    REQUIRE((type == AxtType::F16 || type == AxtType::BF16));
    const auto raw = as_u16();
    for (std::size_t i = 0; i < raw.size(); ++i) {
      out[i] = type == AxtType::F16 ? fp16_to_fp32(raw[i]) : bf16_to_fp32(raw[i]);
    }
    return out;
  }
  std::vector<std::uint16_t> as_u16() const {
    REQUIRE((type == AxtType::F16 || type == AxtType::BF16));
    std::vector<std::uint16_t> out(numel());
    std::memcpy(out.data(), data.data(), out.size() * sizeof(std::uint16_t));
    return out;
  }
  std::vector<std::int32_t> as_i32() const {
    REQUIRE(type == AxtType::I32);
    std::vector<std::int32_t> out(numel());
    std::memcpy(out.data(), data.data(), out.size() * sizeof(std::int32_t));
    return out;
  }
};

// Loads fixtures/kernels/<name>.axt. Every fixture is little-endian and so is every target.
inline FixtureTensor load_axt(const std::string& name) {
  const auto bytes = read_fixture("kernels/" + name + ".axt");
  INFO("fixture " << name);
  REQUIRE(bytes.size() >= 12);
  REQUIRE(std::memcmp(bytes.data(), "AXT1", 4) == 0);
  std::size_t offset = 4;
  const auto read_u32 = [&] {
    std::uint32_t v = 0;
    std::memcpy(&v, bytes.data() + offset, sizeof v);
    offset += sizeof v;
    return v;
  };
  const auto read_u64 = [&] {
    std::uint64_t v = 0;
    std::memcpy(&v, bytes.data() + offset, sizeof v);
    offset += sizeof v;
    return v;
  };
  FixtureTensor t;
  const std::uint32_t ndim = read_u32();
  REQUIRE(ndim <= 4);
  REQUIRE(bytes.size() >= offset + ndim * sizeof(std::uint64_t) + sizeof(std::uint32_t));
  for (std::uint32_t i = 0; i < ndim; ++i) {
    t.dims.push_back(read_u64());
  }
  const std::uint32_t type = read_u32();
  REQUIRE(type <= 3);
  t.type = static_cast<AxtType>(type);
  const std::size_t elem = t.type == AxtType::F16 || t.type == AxtType::BF16 ? 2 : 4;
  REQUIRE(bytes.size() == offset + t.numel() * elem);
  t.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
  return t;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
    worst = d > worst ? d : worst;
  }
  return worst;
}

}  // namespace axon::test
