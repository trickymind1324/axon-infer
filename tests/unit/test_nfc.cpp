#include "tokenizer/nfc.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

namespace {

std::string from_hex(const std::string& hex) {
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

// input hex -> expected NFC hex, produced by Python's unicodedata.normalize("NFC", ...).
const std::vector<std::pair<const char*, const char*>> kCases = {
    {"65cc81", "c3a9"},            // e + acute -> é
    {"6ecc83", "c3b1"},            // n + tilde -> ñ
    {"41cc8a", "c385"},            // A + ring -> Å
    {"e284ab", "c385"},            // Angstrom sign -> Å (singleton)
    {"e1ba9bcca3", "e1ba9bcca3"},  // already composed with a following mark
    {"71cc87cca3", "71cca3cc87"},  // canonical reordering of marks
    {"71cca3cc87", "71cca3cc87"},
    {"e0a598", "e0a495e0a4bc"},        // composition exclusion stays decomposed
    {"e18480e185a1e186a8", "eab081"},  // Hangul L V T -> syllable
    {"ed959ceab880", "ed959ceab880"},  // Hangul syllables unchanged
    {"e18480e185a1", "eab080"},        // Hangul L V -> LV
    {"ed959ce186a8", "ed959ce186a8"},  // 한 (LVT) + ᆨ: cannot attach, stays as is
    {"636166c3a9", "636166c3a9"},      // café already NFC
    {"61cc81cc81", "c3a1cc81"},        // á + extra acute
    {"41cc8acca3", "e1baa0cc8a"},      // ring + dot below reorder and compose
    {"e0bdb3", "e0bdb1e0bdb2"},        // Tibetan exclusion
    {"eab080e186a8", "eab081"},        // LV + T -> LVT
    {"c385", "c385"},
    {"e284a6", "cea9"},            // Ohm sign -> Omega
    {"e1b88acca3", "e1b88ccc87"},  // D with dot above + dot below -> D dot below + dot above
    {"efac81", "efac81"},          // ligature fi is compatibility only: unchanged
    {"cc81", "cc81"},              // lone combining mark
    {"cc8161", "cc8161"},
    {"6d69786564204153434949206f6e6c79", "6d69786564204153434949206f6e6c79"},
    {"c3a9cc81", "c3a9cc81"},
};

}  // namespace

TEST_CASE("NFC normalization matches Python reference cases", "[tokenizer][nfc]") {
  for (const auto& [in, expected] : kCases) {
    INFO("input " << in);
    CHECK(axon::unicode::normalize_nfc(from_hex(in)) == from_hex(expected));
  }
}

TEST_CASE("NFC fast path", "[tokenizer][nfc]") {
  CHECK(axon::unicode::is_nfc_fast_path("plain ascii and latin-1 \xc3\xa9"));
  CHECK_FALSE(axon::unicode::is_nfc_fast_path("e\xcc\x81"));
  CHECK(axon::unicode::normalize_nfc("") == "");
  CHECK(axon::unicode::normalize_nfc("\xff") == "\xef\xbf\xbd");  // invalid byte becomes U+FFFD
}
