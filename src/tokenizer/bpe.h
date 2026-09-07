#pragma once

#include "axon/tokenizer.h"

#include "tokenizer/merge.h"
#include "tokenizer/pretokenizer.h"
#include "tokenizer/vocab.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace axon {

// Byte-level BPE (the "gpt2" vocabulary model): every byte maps to a printable code point,
// chunks from the pre-tokenizer are merged by rank, pieces are looked up in the vocabulary.
class BpeTokenizer final : public Tokenizer {
public:
  static Expected<std::unique_ptr<Tokenizer>> create(std::unique_ptr<Vocab> vocab);

protected:
  void encode_fragment(std::string_view text, bool at_start,
                       std::vector<std::int32_t>& out) const override;
  void append_token_bytes(std::int32_t id, std::string& out) const override;

private:
  BpeTokenizer(std::unique_ptr<Vocab> vocab, PreType pre_type);
  void encode_chunk(std::string_view chunk, std::vector<std::int32_t>& out) const;

  PreType pre_type_;
  std::array<std::string, 256> byte_pieces_;  // byte -> its byte-level code point, UTF-8 encoded
  std::array<std::int32_t, 256> byte_ids_{};  // byte -> id of its single piece, or unk
  std::unordered_map<std::uint32_t, std::uint8_t> cp_to_byte_;
  MergeRanks ranks_;
};

}  // namespace axon
