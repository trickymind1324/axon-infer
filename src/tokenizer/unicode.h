#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// UTF-8 handling and the character classes the pre-tokenizer needs. No locale, no ICU.
namespace axon::unicode {

constexpr std::uint32_t kReplacement = 0xFFFD;

struct Decoded {
  std::uint32_t code_point;  // kReplacement when invalid
  std::size_t length;        // bytes consumed (1 for an invalid lead byte)
  bool valid;
};

// Decodes the sequence starting at text[pos]. pos must be < text.size().
Decoded decode(std::string_view text, std::size_t pos) noexcept;

// Number of bytes a sequence with this lead byte should have; 0 for a byte that cannot lead.
std::size_t sequence_length(unsigned char lead) noexcept;

void append(std::string& out, std::uint32_t code_point);

// Whether `bytes` is a complete, well-formed UTF-8 string.
bool is_valid(std::string_view bytes) noexcept;

bool is_letter(std::uint32_t cp) noexcept;      // \p{L}
bool is_number(std::uint32_t cp) noexcept;      // \p{N}
bool is_whitespace(std::uint32_t cp) noexcept;  // \s (White_Space property)
inline bool is_newline(std::uint32_t cp) noexcept {
  return cp == '\r' || cp == '\n';
}

}  // namespace axon::unicode
