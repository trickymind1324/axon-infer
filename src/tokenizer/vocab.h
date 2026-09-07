#pragma once

#include "axon/tokenizer.h"
#include "core/expected.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace axon {

class GGUFFile;

// The token table read from tokenizer.ggml.* metadata. Pieces are copied into one owned
// buffer so the vocabulary does not depend on the GGUF mapping.
class Vocab {
public:
  static Expected<std::unique_ptr<Vocab>> from_gguf(const GGUFFile& file);

  [[nodiscard]] std::int32_t size() const noexcept {
    return static_cast<std::int32_t>(pieces_.size());
  }
  [[nodiscard]] std::string_view piece(std::int32_t id) const noexcept {
    return pieces_[static_cast<std::size_t>(id)];
  }
  [[nodiscard]] TokenType type(std::int32_t id) const noexcept {
    return types_[static_cast<std::size_t>(id)];
  }
  [[nodiscard]] float score(std::int32_t id) const noexcept {
    return scores_.empty() ? 0.0F : scores_[static_cast<std::size_t>(id)];
  }
  [[nodiscard]] bool has_scores() const noexcept { return !scores_.empty(); }
  [[nodiscard]] std::optional<std::int32_t> find(std::string_view piece) const noexcept;
  [[nodiscard]] bool valid(std::int32_t id) const noexcept { return id >= 0 && id < size(); }
  [[nodiscard]] bool is_special(std::int32_t id) const noexcept {
    const TokenType t = type(id);
    return t == TokenType::Control || t == TokenType::UserDefined;
  }

  [[nodiscard]] const std::vector<std::string_view>& merges() const noexcept { return merges_; }
  // Pieces matched literally in text (control, user-defined, unknown), longest first.
  [[nodiscard]] const std::vector<std::int32_t>& specials() const noexcept { return specials_; }
  [[nodiscard]] const std::vector<std::int32_t>& eog() const noexcept { return eog_; }

  std::int32_t bos = -1;
  std::int32_t eos = -1;
  std::int32_t pad = -1;
  std::int32_t unk = -1;
  bool add_bos = true;
  bool add_space_prefix = true;
  std::string model;
  std::string pre;
  std::string chat_template;

private:
  std::string storage_;  // all piece and merge bytes, back to back
  std::vector<std::string_view> pieces_;
  std::vector<TokenType> types_;
  std::vector<float> scores_;
  std::vector<std::string_view> merges_;
  std::unordered_map<std::string_view, std::int32_t> ids_;
  std::vector<std::int32_t> specials_;
  std::vector<std::int32_t> eog_;
};

}  // namespace axon
