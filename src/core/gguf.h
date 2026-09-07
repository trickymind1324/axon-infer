#pragma once

#include "core/dtype.h"
#include "core/expected.h"
#include "core/os.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace axon {

enum class GGUFValueType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

std::string_view gguf_value_type_name(GGUFValueType type);

// One metadata value. Strings and string arrays are views into the mapped file and stay valid
// for the lifetime of the GGUFFile that produced them.
//
// Arrays of strings, float32 and int32 (vocab, scores, token types) are stored as flat typed
// vectors. Arrays of any other element type, including nested arrays, are stored as a vector
// of MetadataValue and reached through as_array().
class MetadataValue {
public:
  MetadataValue() = default;

  [[nodiscard]] GGUFValueType type() const noexcept { return type_; }
  [[nodiscard]] bool is_array() const noexcept { return type_ == GGUFValueType::Array; }
  [[nodiscard]] GGUFValueType array_element_type() const noexcept { return element_type_; }
  [[nodiscard]] std::size_t array_size() const noexcept;

  // Typed accessors. They never convert silently across kinds: as_uint on an Int32 fails.
  [[nodiscard]] std::optional<std::uint64_t> as_uint() const noexcept;
  [[nodiscard]] std::optional<std::int64_t>
  as_int() const noexcept;  // signed, or unsigned that fits
  [[nodiscard]] std::optional<double> as_float() const noexcept;
  [[nodiscard]] std::optional<bool> as_bool() const noexcept;
  [[nodiscard]] std::optional<std::string_view> as_string() const noexcept;

  [[nodiscard]] std::optional<std::span<const std::string_view>> as_string_array() const noexcept;
  [[nodiscard]] std::optional<std::span<const float>> as_float_array() const noexcept;
  [[nodiscard]] std::optional<std::span<const std::int32_t>> as_int32_array() const noexcept;
  // Generic elements for every other array element type. nullptr for typed arrays and scalars.
  [[nodiscard]] const std::vector<MetadataValue>* as_array() const noexcept;

  // Construction helpers used by the parser and by tests.
  static MetadataValue make_uint(GGUFValueType type, std::uint64_t value);
  static MetadataValue make_int(GGUFValueType type, std::int64_t value);
  static MetadataValue make_float(GGUFValueType type, double value);
  static MetadataValue make_bool(bool value);
  static MetadataValue make_string(std::string_view value);
  static MetadataValue make_string_array(std::vector<std::string_view> values);
  static MetadataValue make_float_array(std::vector<float> values);
  static MetadataValue make_int32_array(std::vector<std::int32_t> values);
  static MetadataValue make_array(GGUFValueType element_type, std::vector<MetadataValue> values);

private:
  GGUFValueType type_ = GGUFValueType::UInt8;
  GGUFValueType element_type_ = GGUFValueType::UInt8;
  std::uint64_t uint_ = 0;
  std::int64_t int_ = 0;
  double float_ = 0.0;
  bool bool_ = false;
  std::string_view string_;
  std::vector<std::string_view> strings_;
  std::vector<float> floats_;
  std::vector<std::int32_t> ints_;
  std::vector<MetadataValue> nested_;
};

struct TensorInfo {
  std::string_view name;  // into the mapped file
  DType dtype = DType::F32;
  std::uint32_t n_dims = 0;
  // shape[0] is the innermost (fastest varying) dimension, exactly as stored in the file.
  // A weight that maps x[n_in] to y[n_out] has shape {n_in, n_out}.
  std::array<std::uint64_t, 4> shape{1, 1, 1, 1};
  std::uint64_t offset = 0;         // relative to the data section
  const std::byte* data = nullptr;  // absolute, into the mapping
  std::size_t nbytes = 0;

  [[nodiscard]] std::uint64_t ne(std::size_t i) const noexcept { return i < 4 ? shape[i] : 1; }
  [[nodiscard]] std::uint64_t n_elements() const noexcept {
    return shape[0] * shape[1] * shape[2] * shape[3];
  }
};

// A parsed, fully validated GGUF v3 file backed by a read-only memory mapping. After open()
// succeeds, no accessor can fail because of malformed data: every offset and length has been
// checked against the file size.
class GGUFFile {
public:
  static Expected<GGUFFile> open(const std::filesystem::path& path);
  // Parses an in-memory image. The bytes must outlive the returned object. Used by tests and
  // the fuzzer; open() uses it on the mapping.
  static Expected<GGUFFile> parse(std::span<const std::byte> bytes);

  GGUFFile(GGUFFile&& other) noexcept;
  GGUFFile& operator=(GGUFFile&& other) noexcept;
  GGUFFile(const GGUFFile&) = delete;
  GGUFFile& operator=(const GGUFFile&) = delete;
  ~GGUFFile();

  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
  [[nodiscard]] std::uint64_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] std::size_t file_size() const noexcept { return bytes_.size(); }
  [[nodiscard]] std::size_t data_offset() const noexcept { return data_offset_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  [[nodiscard]] const MetadataValue* metadata(std::string_view key) const noexcept;
  [[nodiscard]] std::span<const std::pair<std::string_view, MetadataValue>>
  metadata_entries() const noexcept {
    return entries_;
  }

  // Getters that build the "expected X, found Y" message for missing or mistyped keys.
  [[nodiscard]] Expected<std::uint64_t> get_uint(std::string_view key) const;
  [[nodiscard]] Expected<std::int64_t> get_int(std::string_view key) const;
  [[nodiscard]] Expected<double> get_float(std::string_view key) const;
  [[nodiscard]] Expected<bool> get_bool(std::string_view key) const;
  [[nodiscard]] Expected<std::string_view> get_string(std::string_view key) const;
  [[nodiscard]] std::uint64_t get_uint_or(std::string_view key, std::uint64_t fallback) const;
  [[nodiscard]] double get_float_or(std::string_view key, double fallback) const;
  [[nodiscard]] std::string_view get_string_or(std::string_view key,
                                               std::string_view fallback) const;

  [[nodiscard]] const TensorInfo* tensor(std::string_view name) const noexcept;
  [[nodiscard]] std::span<const TensorInfo> tensors() const noexcept { return tensors_; }
  [[nodiscard]] std::size_t tensor_count() const noexcept { return tensors_.size(); }
  [[nodiscard]] std::size_t tensor_data_bytes() const noexcept { return tensor_data_bytes_; }

  [[nodiscard]] std::string_view architecture() const {
    return get_string_or("general.architecture", "");
  }
  [[nodiscard]] std::string_view model_name() const { return get_string_or("general.name", ""); }

private:
  GGUFFile() = default;
  void release() noexcept;

  std::filesystem::path path_;
  os::FileMapping mapping_;  // owned when data != nullptr
  std::span<const std::byte> bytes_;
  std::uint32_t version_ = 0;
  std::uint64_t alignment_ = 32;
  std::size_t data_offset_ = 0;
  std::size_t tensor_data_bytes_ = 0;
  std::vector<std::pair<std::string_view, MetadataValue>> entries_;
  std::unordered_map<std::string_view, std::size_t> entry_index_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string_view, std::size_t> tensor_index_;
};

}  // namespace axon
