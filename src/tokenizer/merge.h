#pragma once

#include "core/function_ref.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace axon {

// One symbol of a word being merged: a byte range of the text plus linked-list neighbours.
struct MergeSymbol {
  std::size_t begin;
  std::size_t end;
  int prev;
  int next;
  [[nodiscard]] bool alive() const noexcept { return end > begin; }
};

// "left right" -> rank, built from tokenizer.ggml.merges.
class MergeRanks {
public:
  explicit MergeRanks(const std::vector<std::string_view>& merges);
  [[nodiscard]] bool empty() const noexcept { return ranks_.empty(); }
  // -1 when the pair is not a merge.
  [[nodiscard]] std::int32_t rank(std::string_view left, std::string_view right) const;

private:
  std::unordered_map<std::string, std::int32_t> ranks_;
};

// Repeatedly merges the adjacent pair with the lowest rank (ties: leftmost) until no pair is a
// merge. `symbols` must cover `text` contiguously; consumed symbols are marked dead and
// unlinked, so iterate with `next` from symbol 0 afterwards.
void merge_by_rank(std::string_view text, std::vector<MergeSymbol>& symbols,
                   const MergeRanks& ranks);

// Same loop driven by a score: the adjacent pair whose concatenation scores highest merges
// first (SentencePiece semantics). `score` returns false when the pair cannot merge.
void merge_by_score(std::string_view text, std::vector<MergeSymbol>& symbols,
                    FunctionRef<bool(std::string_view, float&)> score);

}  // namespace axon
