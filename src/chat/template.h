#pragma once

// Turns a conversation into the prompt text a model was trained on. The tokenizer's own Jinja
// template is used when the interpreter can handle it, otherwise a hard-coded format chosen
// from the vocabulary's special tokens.

#include "axon/tokenizer.h"
#include "core/expected.h"

#include "chat/jinja.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace axon::chat {

struct Message {
  std::string role;
  std::string content;
};

struct RenderOptions {
  std::string date_string;  // overrides the Llama 3 date when set, for stable fixtures
};

enum class Family { ChatML, Llama3, Zephyr, Plain };
std::string_view family_name(Family family);

class ChatTemplate {
public:
  static ChatTemplate from_tokenizer(const Tokenizer& tokenizer);
  // Explicit construction, used by tests and by callers with their own template text.
  static ChatTemplate from_source(std::string_view source, Family fallback, std::string bos,
                                  std::string eos);

  // Renders with the Jinja template, falling back to the family format on a runtime error.
  [[nodiscard]] std::string render(std::span<const Message> messages, bool add_generation_prompt,
                                   const RenderOptions& opts = {}) const;
  // Renders with the Jinja template only. Fails when there is none or it hits a runtime error.
  [[nodiscard]] Expected<std::string> render_jinja(std::span<const Message> messages,
                                                   bool add_generation_prompt,
                                                   const RenderOptions& opts = {}) const;
  [[nodiscard]] std::string render_fallback(std::span<const Message> messages,
                                            bool add_generation_prompt) const;

  // "jinja" when the tokenizer's template parsed, otherwise the fallback family's name.
  [[nodiscard]] std::string_view source() const;
  [[nodiscard]] Family fallback() const noexcept { return fallback_; }
  [[nodiscard]] bool has_jinja() const noexcept { return template_.has_value(); }

private:
  ChatTemplate(std::optional<Template> tmpl, Family fallback, std::string bos, std::string eos);

  std::optional<Template> template_;
  Family fallback_;
  std::string bos_;
  std::string eos_;
};

}  // namespace axon::chat
