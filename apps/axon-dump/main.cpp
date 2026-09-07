// axon-dump: prints the metadata and tensor table of a GGUF file.

#include "core/dtype.h"
#include "core/gguf.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::string path;
  bool json = false;
  bool metadata_only = false;
  bool tensors_only = false;
  bool no_arrays = false;
  std::string filter;
  std::size_t array_preview = 8;
};

std::string human_bytes(std::uint64_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  auto value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  return unit == 0 ? fmt::format("{} B", bytes) : fmt::format("{:.1f} {}", value, units[unit]);
}

std::string json_escape(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        out += fmt::format("\\u{:04x}", c);
      } else {
        out.push_back(static_cast<char>(c));
      }
    }
  }
  out.push_back('"');
  return out;
}

std::string json_number(double value) {
  if (std::isnan(value) || std::isinf(value)) {
    return "null";
  }
  return fmt::format("{}", value);
}

std::string scalar_text(const axon::MetadataValue& value, bool json) {
  if (auto s = value.as_string()) {
    return json_escape(*s);  // quoted in both modes
  }
  if (auto b = value.as_bool()) {
    return *b ? "true" : "false";
  }
  if (auto f = value.as_float()) {
    return json ? json_number(*f) : fmt::format("{}", *f);
  }
  if (auto u = value.as_uint()) {
    return fmt::format("{}", *u);
  }
  if (auto i = value.as_int()) {
    return fmt::format("{}", *i);
  }
  return json ? "null" : "?";
}

// Renders a value. In JSON mode arrays are complete; in text mode they are previewed.
std::string value_text(const axon::MetadataValue& value, const Options& opt) {
  if (!value.is_array()) {
    return scalar_text(value, opt.json);
  }
  const std::size_t n = value.array_size();
  const std::size_t shown =
      opt.json ? n : std::min(n, opt.no_arrays ? std::size_t{0} : opt.array_preview);
  std::vector<std::string> items;
  items.reserve(shown);
  if (auto sa = value.as_string_array()) {
    for (std::size_t i = 0; i < shown; ++i) {
      items.push_back(json_escape((*sa)[i]));
    }
  } else if (auto fa = value.as_float_array()) {
    for (std::size_t i = 0; i < shown; ++i) {
      items.push_back(opt.json ? json_number(static_cast<double>((*fa)[i]))
                               : fmt::format("{}", (*fa)[i]));
    }
  } else if (auto ia = value.as_int32_array()) {
    for (std::size_t i = 0; i < shown; ++i) {
      items.push_back(fmt::format("{}", (*ia)[i]));
    }
  } else if (const auto* generic = value.as_array()) {
    for (std::size_t i = 0; i < shown; ++i) {
      items.push_back(value_text((*generic)[i], opt));
    }
  }
  std::string joined;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      joined += opt.json ? "," : " ";
    }
    joined += items[i];
  }
  if (opt.json) {
    return "[" + joined + "]";
  }
  std::string text =
      fmt::format("[{} x {}]", n, axon::gguf_value_type_name(value.array_element_type()));
  if (shown > 0) {
    text += " " + joined;
    if (shown < n) {
      text += " ...";
    }
  }
  return text;
}

std::string shape_text(const axon::TensorInfo& t, bool json) {
  std::string out = "[";
  for (std::uint32_t d = 0; d < t.n_dims; ++d) {
    if (d != 0) {
      out += json ? "," : ", ";
    }
    out += fmt::format("{}", t.shape[d]);
  }
  return out + "]";
}

bool matches(std::string_view name, const Options& opt) {
  return opt.filter.empty() || name.find(opt.filter) != std::string_view::npos;
}

struct DTypeTotal {
  std::size_t count = 0;
  std::uint64_t bytes = 0;
};

std::map<std::string_view, DTypeTotal> summarize(const axon::GGUFFile& file) {
  std::map<std::string_view, DTypeTotal> totals;
  for (const auto& t : file.tensors()) {
    auto& entry = totals[axon::dtype_name(t.dtype)];
    ++entry.count;
    entry.bytes += t.nbytes;
  }
  return totals;
}

void print_text(const axon::GGUFFile& file, const Options& opt) {
  fmt::print("file: {}\n", file.path().string());
  fmt::print("version {}  alignment {}  tensors {}  metadata {}  size {}  data offset {}\n",
             file.version(), file.alignment(), file.tensor_count(), file.metadata_entries().size(),
             human_bytes(file.file_size()), file.data_offset());
  fmt::print("architecture {}  name {}\n", file.architecture().empty() ? "?" : file.architecture(),
             json_escape(file.model_name()));

  if (!opt.tensors_only) {
    fmt::print("\nmetadata\n");
    std::size_t width = 8;
    for (const auto& [key, value] : file.metadata_entries()) {
      if (matches(key, opt)) {
        width = std::max(width, key.size());
      }
    }
    for (const auto& [key, value] : file.metadata_entries()) {
      if (!matches(key, opt)) {
        continue;
      }
      fmt::print("  {:<{}}  {:<7}  {}\n", key, width, axon::gguf_value_type_name(value.type()),
                 value_text(value, opt));
    }
  }

  if (!opt.metadata_only) {
    fmt::print("\ntensors\n");
    std::size_t width = 4;
    for (const auto& t : file.tensors()) {
      if (matches(t.name, opt)) {
        width = std::max(width, t.name.size());
      }
    }
    fmt::print("  {:>5}  {:<{}}  {:<7}  {:<24}  {:>12}  {:>12}  {:>12}\n", "idx", "name", width,
               "dtype", "shape", "elements", "bytes", "offset");
    std::size_t index = 0;
    for (const auto& t : file.tensors()) {
      if (matches(t.name, opt)) {
        fmt::print("  {:>5}  {:<{}}  {:<7}  {:<24}  {:>12}  {:>12}  {:>12}\n", index, t.name, width,
                   axon::dtype_name(t.dtype), shape_text(t, false), t.n_elements(), t.nbytes,
                   t.offset);
      }
      ++index;
    }

    fmt::print("\nsummary\n");
    const auto totals = summarize(file);
    const double all = static_cast<double>(std::max<std::size_t>(file.tensor_data_bytes(), 1));
    for (const auto& [name, total] : totals) {
      fmt::print("  {:<7}  {:>5} tensors  {:>12} bytes  {:>9}  {:>5.1f}%\n", name, total.count,
                 total.bytes, human_bytes(total.bytes),
                 100.0 * static_cast<double>(total.bytes) / all);
    }
    fmt::print("  {:<7}  {:>5} tensors  {:>12} bytes  {:>9}\n", "total", file.tensor_count(),
               file.tensor_data_bytes(), human_bytes(file.tensor_data_bytes()));
  }
}

void print_json(const axon::GGUFFile& file, const Options& opt) {
  fmt::print("{{\n  \"file\": {},\n  \"version\": {},\n  \"alignment\": {},\n  \"file_size\": {},\n"
             "  \"data_offset\": {},\n  \"tensor_data_bytes\": {},\n  \"architecture\": {},\n  "
             "\"name\": {}",
             json_escape(file.path().string()), file.version(), file.alignment(), file.file_size(),
             file.data_offset(), file.tensor_data_bytes(), json_escape(file.architecture()),
             json_escape(file.model_name()));

  if (!opt.tensors_only) {
    fmt::print(",\n  \"metadata\": {{");
    bool first = true;
    for (const auto& [key, value] : file.metadata_entries()) {
      if (!matches(key, opt)) {
        continue;
      }
      fmt::print("{}\n    {}: {}", first ? "" : ",", json_escape(key), value_text(value, opt));
      first = false;
    }
    fmt::print("\n  }}");
  }

  if (!opt.metadata_only) {
    fmt::print(",\n  \"tensors\": [");
    bool first = true;
    for (const auto& t : file.tensors()) {
      if (!matches(t.name, opt)) {
        continue;
      }
      fmt::print(
          "{}\n    {{\"name\": {}, \"dtype\": {}, \"shape\": {}, \"elements\": {}, \"bytes\": {}, "
          "\"offset\": {}}}",
          first ? "" : ",", json_escape(t.name), json_escape(axon::dtype_name(t.dtype)),
          shape_text(t, true), t.n_elements(), t.nbytes, t.offset);
      first = false;
    }
    fmt::print("\n  ],\n  \"summary\": {{");
    first = true;
    for (const auto& [name, total] : summarize(file)) {
      fmt::print("{}\n    {}: {{\"tensors\": {}, \"bytes\": {}}}", first ? "" : ",",
                 json_escape(name), total.count, total.bytes);
      first = false;
    }
    fmt::print("\n  }}");
  }
  fmt::print("\n}}\n");
}

int run(int argc, char** argv) {
  Options opt;
  CLI::App app{"axon-dump: inspect the metadata and tensors of a GGUF file"};
  app.add_option("file", opt.path, "path to a .gguf file")->required();
  app.add_flag("--json", opt.json, "emit JSON (arrays are printed in full)");
  app.add_flag("--metadata", opt.metadata_only, "only the metadata section");
  app.add_flag("--tensors", opt.tensors_only, "only the tensor table and summary");
  app.add_flag("--no-arrays", opt.no_arrays, "hide array contents in text output");
  app.add_option("--filter", opt.filter, "only keys and tensor names containing this substring");
  app.add_option("--array-preview", opt.array_preview, "array elements shown in text output")
      ->default_val(8);
  CLI11_PARSE(app, argc, argv);

  auto file = axon::GGUFFile::open(opt.path);
  if (!file) {
    fmt::print(stderr, "error: {}\n", file.error().message);
    return 2;
  }
  if (opt.json) {
    print_json(*file, opt);
  } else {
    print_text(*file, opt);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    fmt::print(stderr, "error: {}\n", error.what());
    return 1;
  }
}
