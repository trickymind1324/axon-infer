#include "tokenizer/unicode.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace axon::unicode;

TEST_CASE("utf8 decode of valid sequences", "[tokenizer][unicode]") {
  const std::string text = "aé€😀";
  Decoded d = decode(text, 0);
  CHECK(d.code_point == 'a');
  CHECK(d.length == 1);
  d = decode(text, 1);
  CHECK(d.code_point == 0xE9);
  CHECK(d.length == 2);
  d = decode(text, 3);
  CHECK(d.code_point == 0x20AC);
  CHECK(d.length == 3);
  d = decode(text, 6);
  CHECK(d.code_point == 0x1F600);
  CHECK(d.length == 4);
  CHECK(d.valid);
  CHECK(is_valid(text));
}

TEST_CASE("utf8 decode rejects malformed input one byte at a time", "[tokenizer][unicode]") {
  for (const std::string bad : {"\x80", "\xC0\xAF", "\xE0\x80\x80", "\xED\xA0\x80",
                                "\xF4\x90\x80\x80", "\xF8\x88\x80\x80\x80", "\xE2\x82"}) {
    const Decoded d = decode(bad, 0);
    INFO("input bytes: " << bad.size());
    CHECK_FALSE(d.valid);
    CHECK(d.length == 1);
    CHECK(d.code_point == kReplacement);
    CHECK_FALSE(is_valid(bad));
  }
}

TEST_CASE("utf8 append round-trips", "[tokenizer][unicode]") {
  for (const std::uint32_t cp :
       {0x41U, 0x7FU, 0x80U, 0x7FFU, 0x800U, 0xFFFFU, 0x10000U, 0x10FFFFU}) {
    std::string s;
    append(s, cp);
    const Decoded d = decode(s, 0);
    CHECK(d.valid);
    CHECK(d.code_point == cp);
    CHECK(d.length == s.size());
  }
}

TEST_CASE("character classes", "[tokenizer][unicode]") {
  CHECK(is_letter('a'));
  CHECK(is_letter('Z'));
  CHECK(is_letter(0xE9));    // é
  CHECK(is_letter(0x4E2D));  // 中
  CHECK(is_letter(0x0627));  // Arabic alef
  CHECK(is_letter(0x0915));  // Devanagari ka
  CHECK(is_letter(0x0416));  // Cyrillic zhe
  CHECK_FALSE(is_letter('1'));
  CHECK_FALSE(is_letter(' '));
  CHECK_FALSE(is_letter(0x1F600));  // emoji
  CHECK_FALSE(is_letter(0x0301));   // combining acute (Mn)

  CHECK(is_number('0'));
  CHECK(is_number(0x0663));  // Arabic-Indic three
  CHECK(is_number(0x00BD));  // ½ (No)
  CHECK(is_number(0x2167));  // Roman numeral eight (Nl)
  CHECK_FALSE(is_number('a'));

  CHECK(is_whitespace(' '));
  CHECK(is_whitespace('\t'));
  CHECK(is_whitespace('\n'));
  CHECK(is_whitespace('\r'));
  CHECK(is_whitespace(0xA0));
  CHECK(is_whitespace(0x3000));
  CHECK(is_whitespace(0x2028));
  CHECK_FALSE(is_whitespace(0x200B));  // zero-width space is not White_Space
  CHECK_FALSE(is_whitespace('a'));
}
