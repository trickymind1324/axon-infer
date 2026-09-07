#pragma once

#include "core/expected.h"

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace axon {

// The regex families used by the supported byte-level BPE vocabularies. Each is a hand-written
// matcher reproducing the reference pattern (the reference patterns are quoted in the tests).
enum class PreType {
  Gpt2,    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  Llama3,  // (?i:'s|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
           // ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  Qwen2,   // Llama3 with \p{N} (single digit) and NFC normalisation of the input
};

// Maps tokenizer.ggml.pre to a family. Unknown names fail with "unsupported pre-tokenizer".
Expected<PreType> pre_type_from_name(std::string_view name);
bool pre_type_needs_nfc(PreType type) noexcept;
// tiktoken-derived vocabularies emit a chunk that is itself a token without applying merges.
bool pre_type_ignores_merges(PreType type) noexcept;

using ByteRange = std::pair<std::size_t, std::size_t>;  // [begin, end) byte offsets

// Splits text into chunks exactly as the reference regex would, leftmost-first alternation.
void pretokenize(PreType type, std::string_view text, std::vector<ByteRange>& out);

}  // namespace axon
