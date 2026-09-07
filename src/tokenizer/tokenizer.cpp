#include "axon/tokenizer.h"

#include "core/gguf.h"

#include "tokenizer/bpe.h"
#include "tokenizer/spm.h"
#include "tokenizer/unicode.h"
#include "tokenizer/vocab.h"

#include <fmt/format.h>

namespace axon {

Tokenizer::Tokenizer(std::unique_ptr<Vocab> vocab) : vocab_(std::move(vocab)) {}
Tokenizer::~Tokenizer() = default;

Expected<std::unique_ptr<Tokenizer>> Tokenizer::from_gguf(const GGUFFile& file) {
  auto vocab = Vocab::from_gguf(file);
  if (!vocab) {
    return vocab.error();
  }
  const std::string model = (*vocab)->model;
  if (model == "gpt2") {
    return BpeTokenizer::create(std::move(*vocab));
  }
  if (model == "llama") {
    return SpmTokenizer::create(std::move(*vocab));
  }
  return make_error(
      ErrorCode::Unsupported,
      fmt::format(R"(tokenizer.ggml.model: expected "gpt2" or "llama", found "{}")", model));
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, EncodeOptions options) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size() / 3 + 2);
  if (options.add_bos.value_or(vocab_->add_bos)) {
    out.push_back(vocab_->bos);
  }
  if (!options.parse_special || vocab_->specials().empty()) {
    encode_fragment(text, true, out);
    return out;
  }

  // Split around literal special tokens, longest piece first at every position.
  std::size_t fragment_begin = 0;
  bool at_start = true;
  for (std::size_t pos = 0; pos < text.size();) {
    std::int32_t matched = -1;
    for (const std::int32_t id : vocab_->specials()) {
      if (text.substr(pos).starts_with(vocab_->piece(id))) {
        matched = id;
        break;
      }
    }
    if (matched < 0) {
      ++pos;
      continue;
    }
    if (pos > fragment_begin) {
      encode_fragment(text.substr(fragment_begin, pos - fragment_begin), at_start, out);
    }
    out.push_back(matched);
    at_start = false;
    pos += vocab_->piece(matched).size();
    fragment_begin = pos;
  }
  if (fragment_begin < text.size()) {
    encode_fragment(text.substr(fragment_begin), at_start, out);
  }
  return out;
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids, bool skip_special) const {
  std::string out;
  for (const std::int32_t id : ids) {
    if (!vocab_->valid(id)) {
      continue;
    }
    if (skip_special && vocab_->type(id) == TokenType::Control) {
      continue;
    }
    append_token_bytes(id, out);
  }
  if (strips_leading_space() && !out.empty() && out.front() == ' ') {
    out.erase(0, 1);
  }
  return out;
}

std::string_view Tokenizer::token_text(std::int32_t id) const {
  return vocab_->valid(id) ? vocab_->piece(id) : std::string_view{};
}

std::string Tokenizer::token_bytes(std::int32_t id) const {
  std::string out;
  if (vocab_->valid(id)) {
    append_token_bytes(id, out);
  }
  return out;
}

TokenType Tokenizer::token_type(std::int32_t id) const {
  return vocab_->valid(id) ? vocab_->type(id) : TokenType::Undefined;
}

bool Tokenizer::is_special(std::int32_t id) const {
  return vocab_->valid(id) && vocab_->is_special(id);
}

std::optional<std::int32_t> Tokenizer::token_id(std::string_view piece) const {
  return vocab_->find(piece);
}

std::int32_t Tokenizer::bos() const {
  return vocab_->bos;
}
std::int32_t Tokenizer::eos() const {
  return vocab_->eos;
}
std::optional<std::int32_t> Tokenizer::pad() const {
  return vocab_->pad >= 0 ? std::optional<std::int32_t>(vocab_->pad) : std::nullopt;
}
bool Tokenizer::add_bos_default() const {
  return vocab_->add_bos;
}
std::span<const std::int32_t> Tokenizer::eog_tokens() const {
  return vocab_->eog();
}
std::int32_t Tokenizer::vocab_size() const {
  return vocab_->size();
}
std::string_view Tokenizer::model() const {
  return vocab_->model;
}
std::string_view Tokenizer::pre() const {
  return vocab_->pre;
}
std::string_view Tokenizer::chat_template() const {
  return vocab_->chat_template;
}

// ---------------------------------------------------------------------------------------------

StreamingDecoder::StreamingDecoder(const Tokenizer& tokenizer, bool skip_special)
    : tokenizer_(tokenizer), skip_special_(skip_special) {}

void StreamingDecoder::reset() {
  pending_.clear();
  first_ = true;
}

std::string StreamingDecoder::push(std::int32_t id) {
  if (!tokenizer_.vocab_->valid(id)) {
    return {};
  }
  if (skip_special_ && tokenizer_.vocab_->type(id) == TokenType::Control) {
    return {};
  }
  tokenizer_.append_token_bytes(id, pending_);
  if (first_ && !pending_.empty()) {
    if (tokenizer_.strips_leading_space() && pending_.front() == ' ') {
      pending_.erase(0, 1);
    }
    first_ = false;
  }
  return take_complete();
}

std::string StreamingDecoder::take_complete() {
  // Find the longest prefix made of complete sequences; keep a plausible partial tail.
  std::size_t complete = 0;
  std::size_t pos = 0;
  std::string out;
  while (pos < pending_.size()) {
    const auto lead = static_cast<unsigned char>(pending_[pos]);
    const std::size_t len = unicode::sequence_length(lead);
    if (len == 0) {
      // Invalid lead byte: emit a replacement character and move on.
      out.append(pending_, complete, pos - complete);
      unicode::append(out, unicode::kReplacement);
      ++pos;
      complete = pos;
      continue;
    }
    if (pos + len > pending_.size()) {
      break;  // incomplete sequence at the end: wait for more bytes
    }
    const unicode::Decoded d = unicode::decode(pending_, pos);
    if (!d.valid) {
      out.append(pending_, complete, pos - complete);
      unicode::append(out, unicode::kReplacement);
      ++pos;
      complete = pos;
      continue;
    }
    pos += len;
  }
  out.append(pending_, complete, pos - complete);
  pending_.erase(0, pos);
  return out;
}

std::string StreamingDecoder::flush() {
  std::string out = take_complete();
  if (!pending_.empty()) {
    unicode::append(out, unicode::kReplacement);
    pending_.clear();
  }
  return out;
}

}  // namespace axon
