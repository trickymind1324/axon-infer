#pragma once

#include "core/expected.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace axon {

class GGUFFile;
class Vocab;

// tokenizer.ggml.token_type values.
// The width of tokenizer.ggml.token_type in the file, so a raw value converts without
// truncation and the range check after the cast sees what was read.
// NOLINTNEXTLINE(performance-enum-size)
enum class TokenType : std::int32_t {
  Undefined = 0,
  Normal = 1,
  Unknown = 2,
  Control = 3,
  UserDefined = 4,
  Unused = 5,
  Byte = 6,
};

struct EncodeOptions {
  // nullopt follows tokenizer.ggml.add_bos_token; true/false override it.
  std::optional<bool> add_bos = std::nullopt;
  // Recognise special tokens (control and user-defined pieces) written literally in the text.
  // On by default because that is what Hugging Face tokenizers do; turn it off for untrusted
  // user text that must never inject control tokens.
  bool parse_special = true;
};

// A tokenizer built from GGUF metadata. Two vocabulary models are supported:
//   "gpt2"  byte-level BPE with a pre-tokenizer (SmolLM2, Llama 3, Qwen2)
//   "llama" SentencePiece-style BPE with scores and byte fallback (TinyLlama, Llama 2)
// The tokenizer copies everything it needs, so the GGUFFile may be closed afterwards.
class Tokenizer {
public:
  static Expected<std::unique_ptr<Tokenizer>> from_gguf(const GGUFFile& file);
  virtual ~Tokenizer();

  [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text,
                                                 EncodeOptions options = {}) const;
  [[nodiscard]] std::string decode(std::span<const std::int32_t> ids,
                                   bool skip_special = true) const;

  // The piece exactly as stored in the vocabulary (e.g. "Ġthe" or "▁the").
  [[nodiscard]] std::string_view token_text(std::int32_t id) const;
  // The bytes a token contributes to decoded text (e.g. " the"). May be a partial UTF-8 sequence.
  [[nodiscard]] std::string token_bytes(std::int32_t id) const;
  [[nodiscard]] TokenType token_type(std::int32_t id) const;
  [[nodiscard]] bool is_special(std::int32_t id) const;  // Control or UserDefined
  [[nodiscard]] std::optional<std::int32_t> token_id(std::string_view piece) const;

  [[nodiscard]] std::int32_t bos() const;
  [[nodiscard]] std::int32_t eos() const;
  [[nodiscard]] std::optional<std::int32_t> pad() const;
  [[nodiscard]] bool add_bos_default() const;
  // End-of-generation ids: eos plus other known terminators (eot, <|im_end|>, ...).
  [[nodiscard]] std::span<const std::int32_t> eog_tokens() const;
  [[nodiscard]] std::int32_t vocab_size() const;
  [[nodiscard]] std::string_view model() const;
  [[nodiscard]] std::string_view pre() const;
  [[nodiscard]] std::string_view chat_template() const;

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

protected:
  explicit Tokenizer(std::unique_ptr<Vocab> vocab);
  // Encodes one fragment of ordinary text (no special tokens inside). `at_start` is true for
  // the first fragment of the whole input.
  virtual void encode_fragment(std::string_view text, bool at_start,
                               std::vector<std::int32_t>& out) const = 0;
  virtual void append_token_bytes(std::int32_t id, std::string& out) const = 0;
  // Whether decoded text starts with a synthetic space that must be dropped (SentencePiece).
  [[nodiscard]] virtual bool strips_leading_space() const { return false; }

  [[nodiscard]] const Vocab& vocab() const { return *vocab_; }

private:
  friend class StreamingDecoder;
  std::unique_ptr<Vocab> vocab_;
};

// Accumulates token bytes and releases text only at complete UTF-8 sequence boundaries.
class StreamingDecoder {
public:
  explicit StreamingDecoder(const Tokenizer& tokenizer, bool skip_special = true);
  // Text that is safe to print now; may be empty while a multi-byte sequence is incomplete.
  std::string push(std::int32_t id);
  // Whatever is still buffered at the end of generation (invalid bytes become U+FFFD).
  std::string flush();
  void reset();

private:
  std::string take_complete();

  const Tokenizer& tokenizer_;
  bool skip_special_;
  bool first_ = true;
  std::string pending_;
};

}  // namespace axon
