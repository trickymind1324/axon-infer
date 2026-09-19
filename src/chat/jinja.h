#pragma once

// A small Jinja interpreter, sized from the chat templates that ship in GGUF files. It renders
// the SmolLM2, TinyLlama, Llama 3 and Qwen2 templates with tools left undefined, and reports an
// error for anything outside the subset so the caller can fall back to a hard-coded format.
//
// Supported: text, {{ }}, {% %}, {# #}, the `-` whitespace control on every delimiter, and
// Hugging Face's environment defaults (trim_blocks and lstrip_blocks on). Tags: if, elif, else,
// endif, for x in expr, endfor with loop.first / loop.last / loop.index0 / loop.index /
// loop.length, and set name = expr. Expressions: string, integer, true, false, none, names,
// a.b, a['b'], a[0], a[1:], + on strings and integers, == and !=, and, or, not, parentheses,
// x in y, the tests defined, none, mapping and iterable (with `is not`), the filters trim and
// length, and the calls strftime_now(format) and raise_exception(message).

#include "core/expected.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace axon::chat {

class Value;
using List = std::vector<Value>;
using Map = std::vector<std::pair<std::string, Value>>;  // insertion ordered

class Value {
public:
  enum class Kind : std::uint8_t { Undefined, None, Bool, Int, String, List, Map };

  Value() = default;
  static Value none();
  static Value boolean(bool b);
  static Value integer(std::int64_t i);
  static Value string(std::string s);
  static Value list(List items);
  static Value map(Map entries);

  [[nodiscard]] Kind kind() const noexcept { return kind_; }
  [[nodiscard]] bool truthy() const;
  [[nodiscard]] bool as_bool() const { return bool_; }
  [[nodiscard]] std::int64_t as_int() const { return int_; }
  [[nodiscard]] const std::string& as_string() const { return string_; }
  [[nodiscard]] const List& as_list() const { return list_; }
  [[nodiscard]] const Map& as_map() const { return map_; }
  [[nodiscard]] const Value* get(std::string_view key) const;  // map lookup, nullptr if absent
  [[nodiscard]] bool equals(const Value& other) const;
  [[nodiscard]] std::string_view kind_name() const;

private:
  Kind kind_ = Kind::Undefined;
  bool bool_ = false;
  std::int64_t int_ = 0;
  std::string string_;
  List list_;
  Map map_;
};

struct Node;

class Template {
public:
  static Expected<Template> parse(std::string_view source);
  Template(Template&&) noexcept;
  Template& operator=(Template&&) noexcept;
  ~Template();

  // Renders with the given top-level variables. Fails on a runtime error such as an
  // unsupported filter being reached or raise_exception being called.
  [[nodiscard]] Expected<std::string> render(const Map& variables) const;

private:
  explicit Template(std::vector<std::unique_ptr<Node>> body);
  std::vector<std::unique_ptr<Node>> body_;
};

}  // namespace axon::chat
