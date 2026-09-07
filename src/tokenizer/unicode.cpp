#include "tokenizer/unicode.h"

#include "tokenizer/unicode_tables.h"

#include <algorithm>
#include <iterator>

namespace axon::unicode {

namespace {

bool in_ranges(const unicode_tables::CodePointRange* first,
               const unicode_tables::CodePointRange* last, std::uint32_t cp) noexcept {
  // Binary search for the first range whose `last` is >= cp.
  const auto* it = std::lower_bound(
      first, last, cp,
      [](const unicode_tables::CodePointRange& r, std::uint32_t v) { return r.last < v; });
  return it != last && it->first <= cp;
}

}  // namespace

std::size_t sequence_length(unsigned char lead) noexcept {
  if (lead < 0x80) {
    return 1;
  }
  if ((lead & 0xE0) == 0xC0) {
    return lead >= 0xC2 ? 2 : 0;  // C0 and C1 would be overlong
  }
  if ((lead & 0xF0) == 0xE0) {
    return 3;
  }
  if ((lead & 0xF8) == 0xF0) {
    return lead <= 0xF4 ? 4 : 0;
  }
  return 0;
}

Decoded decode(std::string_view text, std::size_t pos) noexcept {
  const auto lead = static_cast<unsigned char>(text[pos]);
  const std::size_t len = sequence_length(lead);
  if (len == 0 || pos + len > text.size()) {
    return {.code_point = kReplacement, .length = 1, .valid = false};
  }
  if (len == 1) {
    return {.code_point = lead, .length = 1, .valid = true};
  }
  std::uint32_t cp = lead & (0x7FU >> len);
  for (std::size_t i = 1; i < len; ++i) {
    const auto c = static_cast<unsigned char>(text[pos + i]);
    if ((c & 0xC0) != 0x80) {
      return {.code_point = kReplacement, .length = 1, .valid = false};
    }
    cp = (cp << 6) | (c & 0x3FU);
  }
  // Reject overlong encodings, surrogates and out-of-range values.
  const bool overlong = (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000);
  if (overlong || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
    return {.code_point = kReplacement, .length = 1, .valid = false};
  }
  return {.code_point = cp, .length = len, .valid = true};
}

void append(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool is_valid(std::string_view bytes) noexcept {
  std::size_t pos = 0;
  while (pos < bytes.size()) {
    const Decoded d = decode(bytes, pos);
    if (!d.valid) {
      return false;
    }
    pos += d.length;
  }
  return true;
}

bool is_letter(std::uint32_t cp) noexcept {
  return in_ranges(std::begin(unicode_tables::kLetters), std::end(unicode_tables::kLetters), cp);
}

bool is_number(std::uint32_t cp) noexcept {
  return in_ranges(std::begin(unicode_tables::kNumbers), std::end(unicode_tables::kNumbers), cp);
}

bool is_whitespace(std::uint32_t cp) noexcept {
  return in_ranges(std::begin(unicode_tables::kWhiteSpace), std::end(unicode_tables::kWhiteSpace),
                   cp);
}

}  // namespace axon::unicode
