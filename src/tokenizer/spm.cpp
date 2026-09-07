#include "tokenizer/spm.h"

#include "tokenizer/unicode.h"

#include <fmt/format.h>

namespace axon {

namespace {

constexpr std::string_view kMetaSpace = "\xE2\x96\x81";  // U+2581 "▁"

}  // namespace

Expected<std::unique_ptr<Tokenizer>> SpmTokenizer::create(std::unique_ptr<Vocab> vocab) {
  if (!vocab->has_scores() && vocab->merges().empty()) {
    return make_error(ErrorCode::Parse, "tokenizer: SentencePiece vocabulary requires "
                                        "tokenizer.ggml.scores or tokenizer.ggml.merges");
  }
  std::unique_ptr<SpmTokenizer> tokenizer(new SpmTokenizer(std::move(vocab)));
  for (std::uint32_t b = 0; b < 256; ++b) {
    const auto id = tokenizer->vocab().find(fmt::format("<0x{:02X}>", b));
    tokenizer->byte_ids_[b] = id ? *id : -1;
  }
  return std::unique_ptr<Tokenizer>(std::move(tokenizer));
}

SpmTokenizer::SpmTokenizer(std::unique_ptr<Vocab> vocab)
    : Tokenizer(std::move(vocab)), ranks_(this->vocab().merges()) {}

void SpmTokenizer::encode_fragment(std::string_view text, bool at_start,
                                   std::vector<std::int32_t>& out) const {
  if (text.empty()) {
    return;
  }
  // Escape spaces, then prepend the meta space to the first fragment unless it already starts
  // with one (Hugging Face's Metaspace pre-tokenizer with prepend_scheme "first").
  std::string escaped;
  escaped.reserve(text.size() + 4);
  for (const char c : text) {
    if (c == ' ') {
      escaped.append(kMetaSpace);
    } else {
      escaped.push_back(c);
    }
  }
  if (at_start && vocab().add_space_prefix && !escaped.starts_with(kMetaSpace)) {
    escaped.insert(0, kMetaSpace);
  }

  // One symbol per UTF-8 character.
  std::vector<MergeSymbol> symbols;
  for (std::size_t pos = 0; pos < escaped.size();) {
    const unicode::Decoded d = unicode::decode(escaped, pos);
    const int index = static_cast<int>(symbols.size());
    symbols.push_back({.begin = pos, .end = pos + d.length, .prev = index - 1, .next = index + 1});
    pos += d.length;
  }
  symbols.back().next = -1;

  if (!ranks_.empty()) {
    merge_by_rank(escaped, symbols, ranks_);
  } else {
    merge_by_score(escaped, symbols, [&](std::string_view merged, float& score) {
      const auto id = vocab().find(merged);
      if (!id) {
        return false;
      }
      score = vocab().score(*id);
      return true;
    });
  }

  for (int i = 0; i >= 0; i = symbols[static_cast<std::size_t>(i)].next) {
    const MergeSymbol& s = symbols[static_cast<std::size_t>(i)];
    const std::string_view piece = std::string_view(escaped).substr(s.begin, s.end - s.begin);
    if (const auto id = vocab().find(piece)) {
      out.push_back(*id);
      continue;
    }
    for (const char c : piece) {  // byte fallback
      const std::int32_t id = byte_ids_[static_cast<unsigned char>(c)];
      out.push_back(id >= 0 ? id : vocab().unk);
    }
  }
}

void SpmTokenizer::append_token_bytes(std::int32_t id, std::string& out) const {
  const std::string_view piece = vocab().piece(id);
  if (vocab().type(id) == TokenType::Byte && piece.size() == 6 && piece.starts_with("<0x") &&
      piece.ends_with('>')) {
    out.push_back(static_cast<char>(std::stoi(std::string(piece.substr(3, 2)), nullptr, 16)));
    return;
  }
  for (std::size_t pos = 0; pos < piece.size();) {
    if (piece.substr(pos).starts_with(kMetaSpace)) {
      out.push_back(' ');
      pos += kMetaSpace.size();
    } else {
      out.push_back(piece[pos]);
      ++pos;
    }
  }
}

}  // namespace axon
