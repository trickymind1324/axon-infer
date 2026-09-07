#include "tokenizer/nfc.h"

#include "tokenizer/nfc_tables.h"
#include "tokenizer/unicode.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

namespace axon::unicode {

namespace {

constexpr std::uint32_t kHangulSBase = 0xAC00;
constexpr std::uint32_t kHangulLBase = 0x1100;
constexpr std::uint32_t kHangulVBase = 0x1161;
constexpr std::uint32_t kHangulTBase = 0x11A7;
constexpr std::uint32_t kHangulLCount = 19;
constexpr std::uint32_t kHangulVCount = 21;
constexpr std::uint32_t kHangulTCount = 28;
constexpr std::uint32_t kHangulNCount = kHangulVCount * kHangulTCount;
constexpr std::uint32_t kHangulSCount = kHangulLCount * kHangulNCount;

std::uint8_t combining_class(std::uint32_t cp) noexcept {
  const auto* first = std::begin(nfc_tables::kCombiningClasses);
  const auto* last = std::end(nfc_tables::kCombiningClasses);
  const auto* it = std::lower_bound(
      first, last, cp,
      [](const nfc_tables::CombiningClassRange& r, std::uint32_t v) { return r.last < v; });
  return (it != last && it->first <= cp) ? it->ccc : 0;
}

const nfc_tables::Decomposition* find_decomposition(std::uint32_t cp) noexcept {
  const auto* first = std::begin(nfc_tables::kDecompositions);
  const auto* last = std::end(nfc_tables::kDecompositions);
  const auto* it =
      std::lower_bound(first, last, cp, [](const nfc_tables::Decomposition& d, std::uint32_t v) {
        return d.code_point < v;
      });
  return (it != last && it->code_point == cp) ? it : nullptr;
}

std::uint32_t compose_pair(std::uint32_t first, std::uint32_t second) noexcept {
  // Hangul L + V -> LV, LV + T -> LVT.
  if (first >= kHangulLBase && first < kHangulLBase + kHangulLCount && second >= kHangulVBase &&
      second < kHangulVBase + kHangulVCount) {
    return kHangulSBase +
           ((first - kHangulLBase) * kHangulVCount + (second - kHangulVBase)) * kHangulTCount;
  }
  if (first >= kHangulSBase && first < kHangulSBase + kHangulSCount &&
      (first - kHangulSBase) % kHangulTCount == 0 && second > kHangulTBase &&
      second < kHangulTBase + kHangulTCount) {
    return first + (second - kHangulTBase);
  }
  const auto* begin = std::begin(nfc_tables::kCompositions);
  const auto* end = std::end(nfc_tables::kCompositions);
  const auto* it = std::lower_bound(
      begin, end, std::pair{first, second},
      [](const nfc_tables::Composition& c, const std::pair<std::uint32_t, std::uint32_t>& key) {
        return c.first != key.first ? c.first < key.first : c.second < key.second;
      });
  return (it != end && it->first == first && it->second == second) ? it->composite : 0;
}

void decompose_into(std::uint32_t cp, std::vector<std::uint32_t>& out) {
  if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) {
    const std::uint32_t index = cp - kHangulSBase;
    out.push_back(kHangulLBase + index / kHangulNCount);
    out.push_back(kHangulVBase + (index % kHangulNCount) / kHangulTCount);
    if (const std::uint32_t t = index % kHangulTCount; t != 0) {
      out.push_back(kHangulTBase + t);
    }
    return;
  }
  const auto* d = find_decomposition(cp);
  if (d == nullptr) {
    out.push_back(cp);
    return;
  }
  decompose_into(d->first, out);  // the first element may decompose further
  if (d->second != 0) {
    out.push_back(d->second);
  }
}

}  // namespace

bool is_nfc_fast_path(std::string_view text) noexcept {
  // Bytes below 0xCC can only encode code points below U+0300: no decompositions, no
  // combining marks, nothing that composes with anything.
  return std::ranges::all_of(text, [](char c) { return static_cast<unsigned char>(c) < 0xCC; });
}

std::string normalize_nfc(std::string_view text) {
  if (is_nfc_fast_path(text)) {
    return std::string(text);
  }

  // Decompose fully.
  std::vector<std::uint32_t> cps;
  cps.reserve(text.size());
  for (std::size_t pos = 0; pos < text.size();) {
    const Decoded d = decode(text, pos);
    if (d.valid) {
      decompose_into(d.code_point, cps);
    } else {
      cps.push_back(kReplacement);
    }
    pos += d.length;
  }

  // Canonical ordering: stable-sort every run of non-starters by combining class.
  for (std::size_t i = 0; i < cps.size();) {
    if (combining_class(cps[i]) == 0) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < cps.size() && combining_class(cps[j]) != 0) {
      ++j;
    }
    std::stable_sort(
        cps.begin() + static_cast<std::ptrdiff_t>(i), cps.begin() + static_cast<std::ptrdiff_t>(j),
        [](std::uint32_t a, std::uint32_t b) { return combining_class(a) < combining_class(b); });
    i = j;
  }

  // Compose (the canonical composition algorithm).
  std::vector<std::uint32_t> out;
  out.reserve(cps.size());
  std::size_t starter_index = 0;
  bool have_starter = false;
  std::uint8_t last_ccc = 0;
  for (const std::uint32_t cp : cps) {
    const std::uint8_t ccc = combining_class(cp);
    if (have_starter) {
      // Blocked if a character with ccc >= this one's (or a starter) sits in between.
      const bool blocked = (out.size() - 1 != starter_index) && (last_ccc == 0 || last_ccc >= ccc);
      if (!blocked) {
        if (const std::uint32_t composite = compose_pair(out[starter_index], cp); composite != 0) {
          out[starter_index] = composite;
          continue;
        }
      }
    }
    if (ccc == 0) {
      starter_index = out.size();
      have_starter = true;
    }
    last_ccc = ccc;
    out.push_back(cp);
  }

  std::string result;
  result.reserve(text.size());
  for (const std::uint32_t cp : out) {
    append(result, cp);
  }
  return result;
}

}  // namespace axon::unicode
