#include "core/axt.h"

#include "core/check.h"
#include "core/dtype.h"

#include <fmt/format.h>

#include <cstring>
#include <fstream>
#include <iterator>

namespace axon {

namespace {

constexpr std::size_t kMaxDims = 8;

std::size_t element_size(AxtType type) {
  return type == AxtType::F16 || type == AxtType::BF16 ? 2 : 4;
}

template <class T> T read_le(const std::byte* p) {
  T v{};
  std::memcpy(&v, p, sizeof v);
  return v;  // every supported target is little-endian
}

template <class T> void append_le(std::vector<std::byte>& out, T v) {
  std::byte bytes[sizeof v];
  std::memcpy(bytes, &v, sizeof v);
  out.insert(out.end(), bytes, bytes + sizeof v);
}

}  // namespace

std::size_t AxtTensor::numel() const noexcept {
  std::size_t n = 1;
  for (const auto d : dims) {
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

std::vector<float> AxtTensor::as_f32() const {
  AXON_CHECK(type != AxtType::I32, "axt: as_f32 on an i32 tensor");
  std::vector<float> out(numel());
  if (type == AxtType::F32) {
    std::memcpy(out.data(), data.data(), out.size() * sizeof(float));
    return out;
  }
  const auto raw = as_u16();
  for (std::size_t i = 0; i < raw.size(); ++i) {
    out[i] = type == AxtType::F16 ? fp16_to_fp32(raw[i]) : bf16_to_fp32(raw[i]);
  }
  return out;
}

std::vector<std::uint16_t> AxtTensor::as_u16() const {
  AXON_CHECK(type == AxtType::F16 || type == AxtType::BF16, "axt: as_u16 on a 4-byte tensor");
  std::vector<std::uint16_t> out(numel());
  std::memcpy(out.data(), data.data(), out.size() * sizeof(std::uint16_t));
  return out;
}

std::vector<std::int32_t> AxtTensor::as_i32() const {
  AXON_CHECK(type == AxtType::I32, "axt: as_i32 on a non-i32 tensor");
  std::vector<std::int32_t> out(numel());
  std::memcpy(out.data(), data.data(), out.size() * sizeof(std::int32_t));
  return out;
}

Expected<AxtTensor> parse_axt(std::span<const std::byte> bytes) {
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "AXT1", 4) != 0) {
    return make_error(ErrorCode::Parse, "axt: bad magic or truncated header");
  }
  std::size_t offset = 4;
  const auto ndim = read_le<std::uint32_t>(bytes.data() + offset);
  offset += 4;
  if (ndim > kMaxDims) {
    return make_error(ErrorCode::Parse,
                      fmt::format("axt: {} dimensions, at most {} supported", ndim, kMaxDims));
  }
  if (bytes.size() < offset + ndim * sizeof(std::uint64_t) + sizeof(std::uint32_t)) {
    return make_error(ErrorCode::Parse, "axt: truncated header");
  }
  AxtTensor t;
  for (std::uint32_t i = 0; i < ndim; ++i) {
    t.dims.push_back(read_le<std::uint64_t>(bytes.data() + offset));
    offset += sizeof(std::uint64_t);
  }
  const auto type = read_le<std::uint32_t>(bytes.data() + offset);
  offset += sizeof(std::uint32_t);
  if (type > 3) {
    return make_error(ErrorCode::Parse, fmt::format("axt: unknown dtype {}", type));
  }
  t.type = static_cast<AxtType>(type);
  const std::size_t expected = t.numel() * element_size(t.type);
  if (bytes.size() != offset + expected) {
    return make_error(ErrorCode::Parse, fmt::format("axt: expected {} data bytes, found {}",
                                                    expected, bytes.size() - offset));
  }
  t.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
  return t;
}

Expected<AxtTensor> read_axt(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return make_error(ErrorCode::Io, fmt::format("{}: cannot open", path.string()));
  }
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto parsed = parse_axt(std::as_bytes(std::span<const char>(chars)));
  if (!parsed) {
    return make_error(parsed.error().code,
                      fmt::format("{}: {}", path.string(), parsed.error().message));
  }
  return parsed;
}

Expected<std::size_t> write_axt(const std::filesystem::path& path,
                                std::span<const std::uint64_t> dims, AxtType type,
                                std::span<const std::byte> data) {
  std::size_t numel = 1;
  for (const auto d : dims) {
    numel *= static_cast<std::size_t>(d);
  }
  AXON_CHECK(data.size() == numel * element_size(type), "axt: {} bytes for {} elements",
             data.size(), numel);
  std::vector<std::byte> out;
  out.reserve(16 + dims.size() * 8 + data.size());
  out.insert(out.end(), {std::byte{'A'}, std::byte{'X'}, std::byte{'T'}, std::byte{'1'}});
  append_le(out, static_cast<std::uint32_t>(dims.size()));
  for (const auto d : dims) {
    append_le(out, d);
  }
  append_le(out, static_cast<std::uint32_t>(type));
  out.insert(out.end(), data.begin(), data.end());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return make_error(ErrorCode::Io, fmt::format("{}: cannot create", path.string()));
  }
  file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
  if (!file) {
    return make_error(ErrorCode::Io, fmt::format("{}: write failed", path.string()));
  }
  return out.size();
}

Expected<std::size_t> write_axt_f32(const std::filesystem::path& path,
                                    std::span<const std::uint64_t> dims,
                                    std::span<const float> data) {
  return write_axt(path, dims, AxtType::F32, std::as_bytes(data));
}

Expected<std::size_t> write_axt_i32(const std::filesystem::path& path,
                                    std::span<const std::uint64_t> dims,
                                    std::span<const std::int32_t> data) {
  return write_axt(path, dims, AxtType::I32, std::as_bytes(data));
}

}  // namespace axon
