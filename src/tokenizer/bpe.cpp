#include "tokenizer/bpe.h"

#include "tokenizer/nfc.h"
#include "tokenizer/unicode.h"

#include <fmt/format.h>

#include <algorithm>

namespace axon {

namespace {

// GPT-2's byte to code point table: printable Latin-1 bytes map to themselves, the rest to
// U+0100 onwards, in ascending order.
std::array<std::uint32_t, 256> byte_to_code_point_table() {
  std::array<std::uint32_t, 256> table{};
  std::uint32_t next = 256;
  for (std::uint32_t b = 0; b < 256; ++b) {
    const bool printable =
        (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    table[b] = printable ? b : next++;
  }
  return table;
}

}  // namespace

Expected<std::unique_ptr<Tokenizer>> BpeTokenizer::create(std::unique_ptr<Vocab> vocab) {
  auto pre_type = pre_type_from_name(vocab->pre);
  if (!pre_type) {
    return pre_type.error();
  }
  std::unique_ptr<BpeTokenizer> tokenizer(new BpeTokenizer(std::move(vocab), *pre_type));
  const Vocab& v = tokenizer->vocab();

  const auto table = byte_to_code_point_table();
  for (std::uint32_t b = 0; b < 256; ++b) {
    std::string piece;
    unicode::append(piece, table[b]);
    tokenizer->byte_pieces_[b] = piece;
    tokenizer->cp_to_byte_.emplace(table[b], static_cast<std::uint8_t>(b));
    // Some vocabularies omit pieces for a few control bytes; those fall back to unk.
    const auto id = v.find(piece);
    tokenizer->byte_ids_[b] = id ? *id : v.unk;
  }

  return std::unique_ptr<Tokenizer>(std::move(tokenizer));
}

BpeTokenizer::BpeTokenizer(std::unique_ptr<Vocab> vocab, PreType pre_type)
    : Tokenizer(std::move(vocab)), pre_type_(pre_type), ranks_(this->vocab().merges()) {}

void BpeTokenizer::encode_fragment(std::string_view text, bool /*at_start*/,
                                   std::vector<std::int32_t>& out) const {
  std::string normalized;
  if (pre_type_needs_nfc(pre_type_) && !unicode::is_nfc_fast_path(text)) {
    normalized = unicode::normalize_nfc(text);
    text = normalized;
  }
  std::vector<ByteRange> chunks;
  pretokenize(pre_type_, text, chunks);
  for (const auto& [b, e] : chunks) {
    encode_chunk(text.substr(b, e - b), out);
  }
}

void BpeTokenizer::encode_chunk(std::string_view chunk, std::vector<std::int32_t>& out) const {
  // Map bytes to the byte-level alphabet; one symbol per original byte.
  std::string mapped;
  mapped.reserve(chunk.size() * 2);
  std::vector<MergeSymbol> symbols;
  symbols.reserve(chunk.size());
  for (const char c : chunk) {
    const std::string& piece = byte_pieces_[static_cast<unsigned char>(c)];
    const int index = static_cast<int>(symbols.size());
    symbols.push_back({.begin = mapped.size(),
                       .end = mapped.size() + piece.size(),
                       .prev = index - 1,
                       .next = index + 1});
    mapped += piece;
  }
  if (symbols.empty()) {
    return;
  }
  if (pre_type_ignores_merges(pre_type_)) {
    if (const auto id = vocab().find(mapped)) {
      out.push_back(*id);
      return;
    }
  }
  symbols.back().next = -1;
  merge_by_rank(mapped, symbols, ranks_);

  for (int i = 0; i >= 0; i = symbols[static_cast<std::size_t>(i)].next) {
    const MergeSymbol& s = symbols[static_cast<std::size_t>(i)];
    const std::string_view piece = std::string_view(mapped).substr(s.begin, s.end - s.begin);
    if (const auto id = vocab().find(piece)) {
      out.push_back(*id);
    } else {
      // Cannot happen with a well-formed byte-level vocabulary; fall back to single bytes.
      for (std::size_t pos = 0; pos < piece.size();) {
        const unicode::Decoded d = unicode::decode(piece, pos);
        const auto it = cp_to_byte_.find(d.code_point);
        out.push_back(it == cp_to_byte_.end() ? vocab().unk : byte_ids_[it->second]);
        pos += d.length;
      }
    }
  }
}

void BpeTokenizer::append_token_bytes(std::int32_t id, std::string& out) const {
  const std::string_view piece = vocab().piece(id);
  if (vocab().is_special(id)) {
    out.append(piece);  // special pieces are stored as plain text
    return;
  }
  for (std::size_t pos = 0; pos < piece.size();) {
    const unicode::Decoded d = unicode::decode(piece, pos);
    const auto it = cp_to_byte_.find(d.code_point);
    if (it != cp_to_byte_.end()) {
      out.push_back(static_cast<char>(it->second));
    } else {
      out.append(piece.substr(pos, d.length));
    }
    pos += d.length;
  }
}

}  // namespace axon
