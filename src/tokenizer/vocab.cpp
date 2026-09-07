#include "tokenizer/vocab.h"

#include "core/gguf.h"

#include <fmt/format.h>

#include <algorithm>

namespace axon {

namespace {

Error vocab_error(std::string message) {
  return make_error(ErrorCode::Parse, "tokenizer: " + std::move(message));
}

}  // namespace

Expected<std::unique_ptr<Vocab>> Vocab::from_gguf(const GGUFFile& file) {
  auto vocab = std::make_unique<Vocab>();

  auto model = file.get_string("tokenizer.ggml.model");
  if (!model) {
    return vocab_error(model.error().message);
  }
  vocab->model = std::string(*model);
  vocab->pre = std::string(file.get_string_or("tokenizer.ggml.pre", "default"));
  vocab->chat_template = std::string(file.get_string_or("tokenizer.chat_template", ""));

  const MetadataValue* tokens = file.metadata("tokenizer.ggml.tokens");
  const auto token_array = tokens == nullptr ? std::nullopt : tokens->as_string_array();
  if (!token_array) {
    return vocab_error(
        "metadata 'tokenizer.ggml.tokens': expected a string array, found " +
        std::string(tokens == nullptr ? "missing" : gguf_value_type_name(tokens->type())));
  }
  const auto pieces = *token_array;
  if (pieces.empty()) {
    return vocab_error("tokenizer.ggml.tokens: expected at least one token, found none");
  }

  const MetadataValue* types = file.metadata("tokenizer.ggml.token_type");
  std::optional<std::span<const std::int32_t>> type_values;
  if (types != nullptr) {
    type_values = types->as_int32_array();
    if (!type_values) {
      return vocab_error("tokenizer.ggml.token_type: expected an i32 array");
    }
    if (type_values->size() != pieces.size()) {
      return vocab_error(fmt::format(
          "tokenizer.ggml.token_type: expected {} entries to match the tokens, found {}",
          pieces.size(), type_values->size()));
    }
  }

  const MetadataValue* scores = file.metadata("tokenizer.ggml.scores");
  if (scores != nullptr) {
    const auto values = scores->as_float_array();
    if (!values || values->size() != pieces.size()) {
      return vocab_error(
          fmt::format("tokenizer.ggml.scores: expected an f32 array of {} entries", pieces.size()));
    }
    vocab->scores_.assign(values->begin(), values->end());
  }

  std::span<const std::string_view> merges;
  if (const MetadataValue* m = file.metadata("tokenizer.ggml.merges"); m != nullptr) {
    const auto merge_array = m->as_string_array();
    if (!merge_array) {
      return vocab_error("tokenizer.ggml.merges: expected a string array");
    }
    merges = *merge_array;
  }

  // Copy every piece and merge into one buffer, then build views into it.
  std::size_t total = 0;
  for (const auto p : pieces) {
    total += p.size();
  }
  for (const auto m : merges) {
    total += m.size();
  }
  vocab->storage_.reserve(total);
  vocab->pieces_.reserve(pieces.size());
  vocab->types_.reserve(pieces.size());
  vocab->merges_.reserve(merges.size());
  for (const auto p : pieces) {
    vocab->storage_.append(p);
  }
  for (const auto m : merges) {
    vocab->storage_.append(m);
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    const std::string_view view(vocab->storage_.data() + offset, pieces[i].size());
    offset += pieces[i].size();
    vocab->pieces_.push_back(view);
    vocab->types_.push_back(type_values ? static_cast<TokenType>((*type_values)[i])
                                        : TokenType::Normal);
  }
  for (const auto m : merges) {
    vocab->merges_.emplace_back(vocab->storage_.data() + offset, m.size());
    offset += m.size();
  }
  vocab->ids_.reserve(pieces.size() * 2);
  for (std::size_t i = 0; i < vocab->pieces_.size(); ++i) {
    vocab->ids_.emplace(vocab->pieces_[i],
                        static_cast<std::int32_t>(i));  // first wins on duplicates
  }

  auto read_id = [&](const char* key, std::int32_t& out) -> Expected<void> {
    const MetadataValue* v = file.metadata(key);
    if (v == nullptr) {
      return {};
    }
    const auto id = v->as_int();
    if (!id) {
      return vocab_error(
          fmt::format("{}: expected an integer, found {}", key, gguf_value_type_name(v->type())));
    }
    if (*id < 0 || *id >= vocab->size()) {
      return vocab_error(
          fmt::format("{}: expected an id below {}, found {}", key, vocab->size(), *id));
    }
    out = static_cast<std::int32_t>(*id);
    return {};
  };
  for (const auto& [key, field] : {std::pair{"tokenizer.ggml.bos_token_id", &vocab->bos},
                                   std::pair{"tokenizer.ggml.eos_token_id", &vocab->eos},
                                   std::pair{"tokenizer.ggml.padding_token_id", &vocab->pad},
                                   std::pair{"tokenizer.ggml.unknown_token_id", &vocab->unk}}) {
    if (auto r = read_id(key, *field); !r) {
      return r.error();
    }
  }
  if (vocab->bos < 0) {
    return vocab_error("tokenizer.ggml.bos_token_id: expected present, found missing");
  }
  if (vocab->eos < 0) {
    return vocab_error("tokenizer.ggml.eos_token_id: expected present, found missing");
  }

  const bool is_spm = vocab->model == "llama";
  auto read_flag = [&](const char* key, bool fallback) {
    const MetadataValue* v = file.metadata(key);
    return v == nullptr ? fallback : v->as_bool().value_or(fallback);
  };
  // llama.cpp defaults: SentencePiece models add BOS and a space prefix, BPE models do not.
  vocab->add_bos = read_flag("tokenizer.ggml.add_bos_token", is_spm);
  vocab->add_space_prefix = read_flag("tokenizer.ggml.add_space_prefix", is_spm);

  // Tokens matched literally in text (control, user-defined and the unknown token), longest
  // piece first so matching prefers the longest.
  for (std::int32_t id = 0; id < vocab->size(); ++id) {
    const TokenType t = vocab->type(id);
    if ((vocab->is_special(id) || t == TokenType::Unknown) && !vocab->piece(id).empty()) {
      vocab->specials_.push_back(id);
    }
  }
  std::stable_sort(vocab->specials_.begin(), vocab->specials_.end(),
                   [&](std::int32_t a, std::int32_t b) {
                     return vocab->piece(a).size() > vocab->piece(b).size();
                   });

  // End-of-generation set: eos plus the usual chat terminators when they exist as specials.
  vocab->eog_.push_back(vocab->eos);
  for (const char* name :
       {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>"}) {
    if (const auto id = vocab->find(name); id && *id != vocab->eos && vocab->is_special(*id)) {
      vocab->eog_.push_back(*id);
    }
  }
  return vocab;
}

std::optional<std::int32_t> Vocab::find(std::string_view piece) const noexcept {
  const auto it = ids_.find(piece);
  if (it == ids_.end()) {
    return std::nullopt;
  }
  return it->second;
}

}  // namespace axon
