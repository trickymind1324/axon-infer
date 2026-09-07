#include "tokenizer/pretokenizer.h"

#include "tokenizer/unicode.h"

#include <fmt/format.h>

#include <cstdint>

namespace axon {

Expected<PreType> pre_type_from_name(std::string_view name) {
  if (name == "default" || name == "gpt2" || name == "smollm" || name == "olmo" ||
      name == "jina-v2-en") {
    return PreType::Gpt2;
  }
  if (name == "llama-bpe" || name == "llama3") {
    return PreType::Llama3;
  }
  if (name == "qwen2") {
    return PreType::Qwen2;
  }
  return make_error(
      ErrorCode::Unsupported,
      fmt::format(
          "unsupported pre-tokenizer '{}': expected one of default, gpt2, smollm, llama-bpe, qwen2",
          name));
}

bool pre_type_needs_nfc(PreType type) noexcept {
  return type == PreType::Qwen2;
}

bool pre_type_ignores_merges(PreType type) noexcept {
  return type == PreType::Llama3;
}

namespace {

struct Char {
  std::uint32_t cp;
  std::size_t offset;  // byte offset of the code point's first byte
};

enum class Class : std::uint8_t { Letter, Number, Space, Other };

Class classify(std::uint32_t cp) noexcept {
  if (unicode::is_whitespace(cp)) {
    return Class::Space;
  }
  if (unicode::is_letter(cp)) {
    return Class::Letter;
  }
  if (unicode::is_number(cp)) {
    return Class::Number;
  }
  return Class::Other;
}

class Matcher {
public:
  Matcher(PreType type, std::string_view text) : type_(type) {
    chars_.reserve(text.size() + 1);
    for (std::size_t pos = 0; pos < text.size();) {
      const unicode::Decoded d = unicode::decode(text, pos);
      chars_.push_back({d.code_point, pos});
      classes_.push_back(classify(d.code_point));
      pos += d.length;
    }
    chars_.push_back({0, text.size()});  // sentinel: offset of the end
    n_ = classes_.size();
  }

  void run(std::vector<ByteRange>& out) {
    std::size_t i = 0;
    while (i < n_) {
      const std::size_t j = type_ == PreType::Gpt2 ? match_gpt2(i) : match_llama3(i);
      out.emplace_back(chars_[i].offset, chars_[j].offset);
      i = j;
    }
  }

private:
  [[nodiscard]] bool is(std::size_t i, Class c) const noexcept {
    return i < n_ && classes_[i] == c;
  }
  [[nodiscard]] bool is_newline(std::size_t i) const noexcept {
    return i < n_ && unicode::is_newline(chars_[i].cp);
  }

  [[nodiscard]] std::size_t run_of(std::size_t i, Class c) const noexcept {
    while (is(i, c)) {
      ++i;
    }
    return i;
  }

  // 's|'t|'re|'ve|'m|'ll|'d, optionally case-insensitive. Returns end or i if no match.
  [[nodiscard]] std::size_t match_contraction(std::size_t i, bool case_insensitive) const noexcept {
    if (chars_[i].cp != '\'' || i + 1 >= n_) {
      return i;
    }
    auto eq = [&](std::size_t k, char expected) {
      if (k >= n_) {
        return false;
      }
      std::uint32_t cp = chars_[k].cp;
      if (case_insensitive && cp >= 'A' && cp <= 'Z') {
        cp += 'a' - 'A';
      }
      return cp == static_cast<std::uint32_t>(expected);
    };
    if (eq(i + 1, 's') || eq(i + 1, 't') || eq(i + 1, 'm') || eq(i + 1, 'd')) {
      return i + 2;
    }
    if ((eq(i + 1, 'r') && eq(i + 2, 'e')) || (eq(i + 1, 'v') && eq(i + 2, 'e')) ||
        (eq(i + 1, 'l') && eq(i + 2, 'l'))) {
      return i + 3;
    }
    return i;
  }

  // \s+(?!\S): a whitespace run, minus its last character when something follows.
  [[nodiscard]] std::size_t match_space_not_before_nonspace(std::size_t i) const noexcept {
    const std::size_t k = run_of(i, Class::Space);
    if (k == i) {
      return i;
    }
    if (k == n_) {
      return k;
    }
    return k - i >= 2 ? k - 1 : i;
  }

  [[nodiscard]] std::size_t match_gpt2(std::size_t i) const noexcept {
    if (const std::size_t j = match_contraction(i, false); j != i) {
      return j;
    }
    // ' ?\p{L}+', ' ?\p{N}+', ' ?[^\s\p{L}\p{N}]+'
    for (const Class c : {Class::Letter, Class::Number, Class::Other}) {
      std::size_t j = i;
      if (chars_[j].cp == ' ') {
        ++j;
      }
      if (is(j, c)) {
        return run_of(j, c);
      }
    }
    if (const std::size_t j = match_space_not_before_nonspace(i); j != i) {
      return j;
    }
    // \s+ (only reachable for a single whitespace char followed by a non-space, or a run that
    // the previous rule declined), and the guaranteed fallback of one character.
    const std::size_t j = run_of(i, Class::Space);
    return j != i ? j : i + 1;
  }

  [[nodiscard]] std::size_t match_llama3(std::size_t i) const noexcept {
    if (const std::size_t j = match_contraction(i, true); j != i) {
      return j;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    {
      std::size_t j = i;
      if (!is_newline(j) && !is(j, Class::Letter) && !is(j, Class::Number)) {
        ++j;
      }
      if (is(j, Class::Letter)) {
        return run_of(j, Class::Letter);
      }
    }
    // \p{N}{1,3} or \p{N}
    if (is(i, Class::Number)) {
      const std::size_t max = type_ == PreType::Qwen2 ? 1 : 3;
      std::size_t j = i;
      while (j < i + max && is(j, Class::Number)) {
        ++j;
      }
      return j;
    }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    {
      std::size_t j = i;
      if (chars_[j].cp == ' ') {
        ++j;
      }
      if (is(j, Class::Other)) {
        j = run_of(j, Class::Other);
        while (is_newline(j)) {
          ++j;
        }
        return j;
      }
    }
    // \s*[\r\n]+ : the whitespace run through its last newline character
    {
      const std::size_t k = run_of(i, Class::Space);
      std::size_t last_newline = n_;
      for (std::size_t j = i; j < k; ++j) {
        if (unicode::is_newline(chars_[j].cp)) {
          last_newline = j;
        }
      }
      if (last_newline != n_) {
        return last_newline + 1;
      }
    }
    if (const std::size_t j = match_space_not_before_nonspace(i); j != i) {
      return j;
    }
    const std::size_t j = run_of(i, Class::Space);
    return j != i ? j : i + 1;
  }

  PreType type_;
  std::vector<Char> chars_;
  std::vector<Class> classes_;
  std::size_t n_ = 0;
};

}  // namespace

void pretokenize(PreType type, std::string_view text, std::vector<ByteRange>& out) {
  if (text.empty()) {
    return;
  }
  Matcher matcher(type, text);
  matcher.run(out);
}

}  // namespace axon
