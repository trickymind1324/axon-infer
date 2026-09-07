#include "core/gguf.h"

#include "core/byte_reader.h"

#include <fmt/format.h>

#include <limits>
#include <string>

namespace axon {

namespace {

constexpr std::uint32_t kMagic = 0x46554747U;          // "GGUF" little-endian
constexpr std::uint32_t kMagicReversed = 0x47475546U;  // "FUGG": a big-endian file
constexpr std::uint32_t kSupportedVersion = 3;
constexpr std::size_t kMaxStringBytes = std::size_t{16} << 20;
constexpr std::uint32_t kMaxDims = 4;
constexpr std::uint64_t kDefaultAlignment = 32;

Error parse_error(std::string message) {
  return make_error(ErrorCode::Parse, std::move(message));
}

constexpr bool is_power_of_two(std::uint64_t v) {
  return v != 0 && (v & (v - 1)) == 0;
}

// Smallest encoded size of one element of the given type, used to reject array counts that
// cannot possibly fit in the remaining bytes before anything is allocated.
std::size_t min_encoded_size(GGUFValueType type) {
  switch (type) {
  case GGUFValueType::UInt8:
  case GGUFValueType::Int8:
  case GGUFValueType::Bool:
    return 1;
  case GGUFValueType::UInt16:
  case GGUFValueType::Int16:
    return 2;
  case GGUFValueType::UInt32:
  case GGUFValueType::Int32:
  case GGUFValueType::Float32:
    return 4;
  case GGUFValueType::UInt64:
  case GGUFValueType::Int64:
  case GGUFValueType::Float64:
  case GGUFValueType::String:  // u64 length + 0 bytes
    return 8;
  case GGUFValueType::Array:  // u32 type + u64 count
    return 12;
  }
  return 0;
}

bool valid_value_type(std::uint32_t raw) {
  return raw <= static_cast<std::uint32_t>(GGUFValueType::Float64);
}

Expected<MetadataValue> read_scalar(ByteReader& reader, GGUFValueType type, std::string_view what) {
  switch (type) {
  case GGUFValueType::UInt8: {
    auto v = reader.u8(what);
    if (!v)
      return v.error();
    return MetadataValue::make_uint(type, *v);
  }
  case GGUFValueType::Int8: {
    auto v = reader.i8(what);
    if (!v)
      return v.error();
    return MetadataValue::make_int(type, *v);
  }
  case GGUFValueType::UInt16: {
    auto v = reader.u16(what);
    if (!v)
      return v.error();
    return MetadataValue::make_uint(type, *v);
  }
  case GGUFValueType::Int16: {
    auto v = reader.i16(what);
    if (!v)
      return v.error();
    return MetadataValue::make_int(type, *v);
  }
  case GGUFValueType::UInt32: {
    auto v = reader.u32(what);
    if (!v)
      return v.error();
    return MetadataValue::make_uint(type, *v);
  }
  case GGUFValueType::Int32: {
    auto v = reader.i32(what);
    if (!v)
      return v.error();
    return MetadataValue::make_int(type, *v);
  }
  case GGUFValueType::Float32: {
    auto v = reader.f32(what);
    if (!v)
      return v.error();
    return MetadataValue::make_float(type, static_cast<double>(*v));
  }
  case GGUFValueType::Bool: {
    auto v = reader.u8(what);
    if (!v)
      return v.error();
    if (*v > 1) {
      return parse_error(fmt::format("{}: expected a bool of 0 or 1, found {}", what, *v));
    }
    return MetadataValue::make_bool(*v != 0);
  }
  case GGUFValueType::String: {
    auto v = reader.string(what, kMaxStringBytes);
    if (!v)
      return v.error();
    return MetadataValue::make_string(*v);
  }
  case GGUFValueType::UInt64: {
    auto v = reader.u64(what);
    if (!v)
      return v.error();
    return MetadataValue::make_uint(type, *v);
  }
  case GGUFValueType::Int64: {
    auto v = reader.i64(what);
    if (!v)
      return v.error();
    return MetadataValue::make_int(type, *v);
  }
  case GGUFValueType::Float64: {
    auto v = reader.f64(what);
    if (!v)
      return v.error();
    return MetadataValue::make_float(type, *v);
  }
  case GGUFValueType::Array:
    break;
  }
  return parse_error(fmt::format("{}: unexpected array where a scalar was required", what));
}

Expected<MetadataValue> read_value(ByteReader& reader, GGUFValueType type, std::string_view what,
                                   int depth);

Expected<MetadataValue> read_array(ByteReader& reader, std::string_view what, int depth) {
  if (depth > 4) {
    return parse_error(fmt::format("{}: arrays nested deeper than 4 levels", what));
  }
  const std::size_t start = reader.offset();
  auto raw_type = reader.u32(what);
  if (!raw_type)
    return raw_type.error();
  if (!valid_value_type(*raw_type)) {
    return parse_error(
        fmt::format("{} at offset {:#x}: unknown array element type {}", what, start, *raw_type));
  }
  const auto element_type = static_cast<GGUFValueType>(*raw_type);
  auto count = reader.u64(what);
  if (!count)
    return count.error();
  const std::size_t min_size = min_encoded_size(element_type);
  if (*count > reader.remaining() / min_size) {
    return parse_error(
        fmt::format("{} at offset {:#x}: array of {} x {} cannot fit in the {} bytes left", what,
                    start, *count, gguf_value_type_name(element_type), reader.remaining()));
  }
  const auto n = static_cast<std::size_t>(*count);

  switch (element_type) {
  case GGUFValueType::String: {
    std::vector<std::string_view> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto s = reader.string(what, kMaxStringBytes);
      if (!s)
        return s.error();
      values.push_back(*s);
    }
    return MetadataValue::make_string_array(std::move(values));
  }
  case GGUFValueType::Float32: {
    std::vector<float> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto f = reader.f32(what);
      if (!f)
        return f.error();
      values.push_back(*f);
    }
    return MetadataValue::make_float_array(std::move(values));
  }
  case GGUFValueType::Int32: {
    std::vector<std::int32_t> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto v = reader.i32(what);
      if (!v)
        return v.error();
      values.push_back(*v);
    }
    return MetadataValue::make_int32_array(std::move(values));
  }
  default: {
    std::vector<MetadataValue> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto v = read_value(reader, element_type, what, depth + 1);
      if (!v)
        return v.error();
      values.push_back(std::move(*v));
    }
    return MetadataValue::make_array(element_type, std::move(values));
  }
  }
}

Expected<MetadataValue> read_value(ByteReader& reader, GGUFValueType type, std::string_view what,
                                   int depth) {
  if (type == GGUFValueType::Array) {
    return read_array(reader, what, depth);
  }
  return read_scalar(reader, type, what);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// MetadataValue

std::string_view gguf_value_type_name(GGUFValueType type) {
  switch (type) {
  case GGUFValueType::UInt8:
    return "u8";
  case GGUFValueType::Int8:
    return "i8";
  case GGUFValueType::UInt16:
    return "u16";
  case GGUFValueType::Int16:
    return "i16";
  case GGUFValueType::UInt32:
    return "u32";
  case GGUFValueType::Int32:
    return "i32";
  case GGUFValueType::Float32:
    return "f32";
  case GGUFValueType::Bool:
    return "bool";
  case GGUFValueType::String:
    return "string";
  case GGUFValueType::Array:
    return "array";
  case GGUFValueType::UInt64:
    return "u64";
  case GGUFValueType::Int64:
    return "i64";
  case GGUFValueType::Float64:
    return "f64";
  }
  return "unknown";
}

std::size_t MetadataValue::array_size() const noexcept {
  if (!is_array()) {
    return 0;
  }
  switch (element_type_) {
  case GGUFValueType::String:
    return strings_.size();
  case GGUFValueType::Float32:
    return floats_.size();
  case GGUFValueType::Int32:
    return ints_.size();
  default:
    return nested_.size();
  }
}

std::optional<std::uint64_t> MetadataValue::as_uint() const noexcept {
  switch (type_) {
  case GGUFValueType::UInt8:
  case GGUFValueType::UInt16:
  case GGUFValueType::UInt32:
  case GGUFValueType::UInt64:
    return uint_;
  default:
    return std::nullopt;
  }
}

std::optional<std::int64_t> MetadataValue::as_int() const noexcept {
  switch (type_) {
  case GGUFValueType::Int8:
  case GGUFValueType::Int16:
  case GGUFValueType::Int32:
  case GGUFValueType::Int64:
    return int_;
  case GGUFValueType::UInt8:
  case GGUFValueType::UInt16:
  case GGUFValueType::UInt32:
    return static_cast<std::int64_t>(uint_);
  case GGUFValueType::UInt64:
    if (uint_ <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return static_cast<std::int64_t>(uint_);
    }
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

std::optional<double> MetadataValue::as_float() const noexcept {
  if (type_ == GGUFValueType::Float32 || type_ == GGUFValueType::Float64) {
    return float_;
  }
  return std::nullopt;
}

std::optional<bool> MetadataValue::as_bool() const noexcept {
  if (type_ == GGUFValueType::Bool) {
    return bool_;
  }
  return std::nullopt;
}

std::optional<std::string_view> MetadataValue::as_string() const noexcept {
  if (type_ == GGUFValueType::String) {
    return string_;
  }
  return std::nullopt;
}

std::optional<std::span<const std::string_view>> MetadataValue::as_string_array() const noexcept {
  if (is_array() && element_type_ == GGUFValueType::String) {
    return std::span<const std::string_view>(strings_);
  }
  return std::nullopt;
}

std::optional<std::span<const float>> MetadataValue::as_float_array() const noexcept {
  if (is_array() && element_type_ == GGUFValueType::Float32) {
    return std::span<const float>(floats_);
  }
  return std::nullopt;
}

std::optional<std::span<const std::int32_t>> MetadataValue::as_int32_array() const noexcept {
  if (is_array() && element_type_ == GGUFValueType::Int32) {
    return std::span<const std::int32_t>(ints_);
  }
  return std::nullopt;
}

const std::vector<MetadataValue>* MetadataValue::as_array() const noexcept {
  if (is_array() && element_type_ != GGUFValueType::String &&
      element_type_ != GGUFValueType::Float32 && element_type_ != GGUFValueType::Int32) {
    return &nested_;
  }
  return nullptr;
}

MetadataValue MetadataValue::make_uint(GGUFValueType type, std::uint64_t value) {
  MetadataValue v;
  v.type_ = type;
  v.uint_ = value;
  return v;
}

MetadataValue MetadataValue::make_int(GGUFValueType type, std::int64_t value) {
  MetadataValue v;
  v.type_ = type;
  v.int_ = value;
  return v;
}

MetadataValue MetadataValue::make_float(GGUFValueType type, double value) {
  MetadataValue v;
  v.type_ = type;
  v.float_ = value;
  return v;
}

MetadataValue MetadataValue::make_bool(bool value) {
  MetadataValue v;
  v.type_ = GGUFValueType::Bool;
  v.bool_ = value;
  return v;
}

MetadataValue MetadataValue::make_string(std::string_view value) {
  MetadataValue v;
  v.type_ = GGUFValueType::String;
  v.string_ = value;
  return v;
}

MetadataValue MetadataValue::make_string_array(std::vector<std::string_view> values) {
  MetadataValue v;
  v.type_ = GGUFValueType::Array;
  v.element_type_ = GGUFValueType::String;
  v.strings_ = std::move(values);
  return v;
}

MetadataValue MetadataValue::make_float_array(std::vector<float> values) {
  MetadataValue v;
  v.type_ = GGUFValueType::Array;
  v.element_type_ = GGUFValueType::Float32;
  v.floats_ = std::move(values);
  return v;
}

MetadataValue MetadataValue::make_int32_array(std::vector<std::int32_t> values) {
  MetadataValue v;
  v.type_ = GGUFValueType::Array;
  v.element_type_ = GGUFValueType::Int32;
  v.ints_ = std::move(values);
  return v;
}

MetadataValue MetadataValue::make_array(GGUFValueType element_type,
                                        std::vector<MetadataValue> values) {
  MetadataValue v;
  v.type_ = GGUFValueType::Array;
  v.element_type_ = element_type;
  v.nested_ = std::move(values);
  return v;
}

// ---------------------------------------------------------------------------------------------
// GGUFFile

GGUFFile::GGUFFile(GGUFFile&& other) noexcept {
  *this = std::move(other);
}

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    mapping_ = other.mapping_;
    bytes_ = other.bytes_;
    version_ = other.version_;
    alignment_ = other.alignment_;
    data_offset_ = other.data_offset_;
    tensor_data_bytes_ = other.tensor_data_bytes_;
    entries_ = std::move(other.entries_);
    entry_index_ = std::move(other.entry_index_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
    other.mapping_ = {};
    other.bytes_ = {};
  }
  return *this;
}

GGUFFile::~GGUFFile() {
  release();
}

void GGUFFile::release() noexcept {
  entries_.clear();
  entry_index_.clear();
  tensors_.clear();
  tensor_index_.clear();
  os::unmap_file(mapping_);
  bytes_ = {};
}

Expected<GGUFFile> GGUFFile::open(const std::filesystem::path& path) {
  auto mapping = os::map_file(path);
  if (!mapping) {
    return mapping.error();
  }
  auto parsed = parse(std::span<const std::byte>(mapping->data, mapping->size));
  if (!parsed) {
    os::unmap_file(*mapping);
    return make_error(parsed.error().code,
                      fmt::format("{}: {}", path.string(), parsed.error().message));
  }
  parsed->mapping_ = *mapping;
  parsed->path_ = path;
  return parsed;
}

Expected<GGUFFile> GGUFFile::parse(std::span<const std::byte> bytes) {
  GGUFFile file;
  file.bytes_ = bytes;
  ByteReader reader(bytes);

  // Header.
  auto magic = reader.u32("gguf magic");
  if (!magic)
    return magic.error();
  if (*magic == kMagicReversed) {
    return parse_error("gguf magic: file is big-endian, which is not supported");
  }
  if (*magic != kMagic) {
    return parse_error(
        fmt::format("gguf magic: expected \"GGUF\" ({:#010x}), found {:#010x}", kMagic, *magic));
  }
  auto version = reader.u32("gguf version");
  if (!version)
    return version.error();
  if (*version != kSupportedVersion) {
    return parse_error(
        fmt::format("gguf version: expected {}, found {}", kSupportedVersion, *version));
  }
  file.version_ = *version;

  auto n_tensors = reader.u64("gguf tensor count");
  if (!n_tensors)
    return n_tensors.error();
  auto n_kv = reader.u64("gguf metadata count");
  if (!n_kv)
    return n_kv.error();
  // Each tensor record is at least 8 (name) + 4 (dims) + 8 (one dim) + 4 (type) + 8 (offset)
  // bytes; each kv record at least 8 (key) + 4 (type) + 1 (value).
  if (*n_tensors > reader.remaining() / 32) {
    return parse_error(
        fmt::format("gguf tensor count {} cannot fit in a {} byte file", *n_tensors, bytes.size()));
  }
  if (*n_kv > reader.remaining() / 13) {
    return parse_error(
        fmt::format("gguf metadata count {} cannot fit in a {} byte file", *n_kv, bytes.size()));
  }

  // Metadata.
  file.entries_.reserve(static_cast<std::size_t>(*n_kv));
  for (std::uint64_t i = 0; i < *n_kv; ++i) {
    const std::string what = fmt::format("metadata entry {}", i);
    auto key = reader.string(what + " key", kMaxStringBytes);
    if (!key)
      return key.error();
    if (key->empty()) {
      return parse_error(fmt::format("{}: key is empty", what));
    }
    if (file.entry_index_.contains(*key)) {
      return parse_error(fmt::format("{}: duplicate key '{}'", what, *key));
    }
    auto raw_type = reader.u32(what + " type");
    if (!raw_type)
      return raw_type.error();
    if (!valid_value_type(*raw_type)) {
      return parse_error(fmt::format("{} ('{}'): unknown value type {}", what, *key, *raw_type));
    }
    const std::string value_what = fmt::format("metadata '{}'", *key);
    auto value = read_value(reader, static_cast<GGUFValueType>(*raw_type), value_what, 0);
    if (!value)
      return value.error();
    file.entry_index_.emplace(*key, file.entries_.size());
    file.entries_.emplace_back(*key, std::move(*value));
  }

  // Alignment.
  if (const MetadataValue* align = file.metadata("general.alignment"); align != nullptr) {
    if (align->type() != GGUFValueType::UInt32) {
      return parse_error(fmt::format("general.alignment: expected u32, found {}",
                                     gguf_value_type_name(align->type())));
    }
    const std::uint64_t a = align->as_uint().value_or(0);
    if (!is_power_of_two(a)) {
      return parse_error(fmt::format("general.alignment: expected a power of two, found {}", a));
    }
    file.alignment_ = a;
  } else {
    file.alignment_ = kDefaultAlignment;
  }

  // Tensor table.
  file.tensors_.reserve(static_cast<std::size_t>(*n_tensors));
  for (std::uint64_t i = 0; i < *n_tensors; ++i) {
    const std::string what = fmt::format("tensor {}", i);
    TensorInfo info;
    auto name = reader.string(what + " name", kMaxStringBytes);
    if (!name)
      return name.error();
    if (name->empty()) {
      return parse_error(fmt::format("{}: name is empty", what));
    }
    if (file.tensor_index_.contains(*name)) {
      return parse_error(fmt::format("{}: duplicate tensor name '{}'", what, *name));
    }
    info.name = *name;
    const std::string named = fmt::format("tensor '{}'", *name);

    auto n_dims = reader.u32(named + " dimension count");
    if (!n_dims)
      return n_dims.error();
    if (*n_dims == 0 || *n_dims > kMaxDims) {
      return parse_error(
          fmt::format("{}: expected 1 to {} dimensions, found {}", named, kMaxDims, *n_dims));
    }
    info.n_dims = *n_dims;
    std::uint64_t n_elements = 1;
    for (std::uint32_t d = 0; d < *n_dims; ++d) {
      auto dim = reader.u64(fmt::format("{} dimension {}", named, d));
      if (!dim)
        return dim.error();
      if (*dim == 0) {
        return parse_error(fmt::format("{}: dimension {} is zero", named, d));
      }
      if (*dim > std::numeric_limits<std::uint64_t>::max() / n_elements) {
        return parse_error(fmt::format("{}: element count overflows", named));
      }
      n_elements *= *dim;
      info.shape[d] = *dim;
    }

    auto raw_type = reader.u32(named + " type");
    if (!raw_type)
      return raw_type.error();
    const DTypeInfo* dtype = dtype_info(*raw_type);
    if (dtype == nullptr) {
      return parse_error(fmt::format("{}: unknown tensor type id {}", named, *raw_type));
    }
    info.dtype = dtype->type;

    auto offset = reader.u64(named + " offset");
    if (!offset)
      return offset.error();
    info.offset = *offset;

    auto row = row_bytes(info.dtype, info.shape[0]);
    if (!row) {
      return parse_error(fmt::format("{}: {}", named, row.error().message));
    }
    const std::uint64_t n_rows = n_elements / info.shape[0];
    if (*row != 0 && n_rows > std::numeric_limits<std::size_t>::max() / *row) {
      return parse_error(fmt::format("{}: byte size overflows", named));
    }
    info.nbytes = static_cast<std::size_t>(n_rows) * *row;

    file.tensor_index_.emplace(info.name, file.tensors_.size());
    file.tensors_.push_back(info);
  }

  // Data section: aligned to `alignment` after the tensor table.
  const std::uint64_t header_end = reader.offset();
  const std::uint64_t data_offset =
      (header_end + file.alignment_ - 1) / file.alignment_ * file.alignment_;
  if (data_offset > bytes.size()) {
    return parse_error(fmt::format("data section: starts at offset {} beyond the {} byte file",
                                   data_offset, bytes.size()));
  }
  file.data_offset_ = static_cast<std::size_t>(data_offset);
  const std::uint64_t data_size = bytes.size() - data_offset;

  std::size_t total = 0;
  for (TensorInfo& info : file.tensors_) {
    if (info.offset % file.alignment_ != 0) {
      return parse_error(fmt::format("tensor '{}': offset {} is not a multiple of the alignment {}",
                                     info.name, info.offset, file.alignment_));
    }
    if (info.offset > data_size || info.nbytes > data_size - info.offset) {
      return parse_error(
          fmt::format("tensor '{}': {} bytes at data offset {} exceed the {} byte data section",
                      info.name, info.nbytes, info.offset, data_size));
    }
    info.data = bytes.data() + file.data_offset_ + info.offset;
    total += info.nbytes;
  }
  file.tensor_data_bytes_ = total;

  return file;
}

const MetadataValue* GGUFFile::metadata(std::string_view key) const noexcept {
  const auto it = entry_index_.find(key);
  return it == entry_index_.end() ? nullptr : &entries_[it->second].second;
}

namespace {

Error missing(std::string_view key) {
  return make_error(ErrorCode::Parse,
                    fmt::format("metadata '{}': expected present, found missing", key));
}

Error mistyped(std::string_view key, std::string_view expected, GGUFValueType found) {
  return make_error(ErrorCode::Parse, fmt::format("metadata '{}': expected {}, found {}", key,
                                                  expected, gguf_value_type_name(found)));
}

}  // namespace

Expected<std::uint64_t> GGUFFile::get_uint(std::string_view key) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return missing(key);
  if (auto u = v->as_uint())
    return *u;
  return mistyped(key, "an unsigned integer", v->type());
}

Expected<std::int64_t> GGUFFile::get_int(std::string_view key) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return missing(key);
  if (auto i = v->as_int())
    return *i;
  return mistyped(key, "an integer", v->type());
}

Expected<double> GGUFFile::get_float(std::string_view key) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return missing(key);
  if (auto f = v->as_float())
    return *f;
  return mistyped(key, "a float", v->type());
}

Expected<bool> GGUFFile::get_bool(std::string_view key) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return missing(key);
  if (auto b = v->as_bool())
    return *b;
  return mistyped(key, "a bool", v->type());
}

Expected<std::string_view> GGUFFile::get_string(std::string_view key) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return missing(key);
  if (auto s = v->as_string())
    return *s;
  return mistyped(key, "a string", v->type());
}

std::uint64_t GGUFFile::get_uint_or(std::string_view key, std::uint64_t fallback) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return fallback;
  return v->as_uint().value_or(fallback);
}

double GGUFFile::get_float_or(std::string_view key, double fallback) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return fallback;
  return v->as_float().value_or(fallback);
}

std::string_view GGUFFile::get_string_or(std::string_view key, std::string_view fallback) const {
  const MetadataValue* v = metadata(key);
  if (v == nullptr)
    return fallback;
  return v->as_string().value_or(fallback);
}

const TensorInfo* GGUFFile::tensor(std::string_view name) const noexcept {
  const auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace axon
