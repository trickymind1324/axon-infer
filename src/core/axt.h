#pragma once

// The .axt tensor container shared by test fixtures and activation dumps. Little-endian
// throughout: magic "AXT1", u32 ndim, u64 dims[ndim], u32 dtype, raw row-major data.

#include "core/expected.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace axon {

enum class AxtType : std::uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3 };

struct AxtTensor {
  std::vector<std::uint64_t> dims;
  AxtType type = AxtType::F32;
  std::vector<std::byte> data;

  [[nodiscard]] std::size_t numel() const noexcept;
  [[nodiscard]] std::size_t dim(std::size_t i) const {
    return static_cast<std::size_t>(dims.at(i));
  }
  // Element values as float; F16 and BF16 convert exactly. Not valid for I32.
  [[nodiscard]] std::vector<float> as_f32() const;
  [[nodiscard]] std::vector<std::uint16_t> as_u16() const;  // raw F16 or BF16 bits
  [[nodiscard]] std::vector<std::int32_t> as_i32() const;
};

Expected<AxtTensor> parse_axt(std::span<const std::byte> bytes);
Expected<AxtTensor> read_axt(const std::filesystem::path& path);

// Writes one tensor; returns the number of bytes written.
Expected<std::size_t> write_axt(const std::filesystem::path& path,
                                std::span<const std::uint64_t> dims, AxtType type,
                                std::span<const std::byte> data);
Expected<std::size_t> write_axt_f32(const std::filesystem::path& path,
                                    std::span<const std::uint64_t> dims,
                                    std::span<const float> data);
Expected<std::size_t> write_axt_i32(const std::filesystem::path& path,
                                    std::span<const std::uint64_t> dims,
                                    std::span<const std::int32_t> data);

}  // namespace axon
