#pragma once

#include "axon/tokenizer.h"

#include "tokenizer/merge.h"
#include "tokenizer/vocab.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace axon {

// SentencePiece-style BPE (the "llama" vocabulary model): spaces become U+2581, pieces merge
// by the merge table when the file carries one (what Hugging Face's tokenizer does) or by
// vocabulary score otherwise, and unknown characters fall back to <0xNN> byte tokens.
class SpmTokenizer final : public Tokenizer {
public:
  static Expected<std::unique_ptr<Tokenizer>> create(std::unique_ptr<Vocab> vocab);

protected:
  void encode_fragment(std::string_view text, bool at_start,
                       std::vector<std::int32_t>& out) const override;
  void append_token_bytes(std::int32_t id, std::string& out) const override;
  [[nodiscard]] bool strips_leading_space() const override { return vocab().add_space_prefix; }

private:
  explicit SpmTokenizer(std::unique_ptr<Vocab> vocab);

  std::array<std::int32_t, 256> byte_ids_{};  // <0xNN> ids, -1 when absent
  MergeRanks ranks_;
};

}  // namespace axon
