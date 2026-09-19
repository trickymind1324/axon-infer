#include "chat/template.h"

#include "core/log.h"

#include <fmt/format.h>

#include <cctype>

namespace axon::chat {

std::string_view family_name(Family family) {
  switch (family) {
  case Family::ChatML:
    return "chatml";
  case Family::Llama3:
    return "llama3";
  case Family::Zephyr:
    return "zephyr";
  case Family::Plain:
    return "plain";
  }
  return "?";
}

namespace {

// ChatML and Llama 3 mark turns with dedicated tokens. Zephyr-style models such as TinyLlama
// spell their markers as plain text, so the template source is the only place to find them.
Family detect_family(const Tokenizer& tokenizer) {
  if (tokenizer.token_id("<|im_start|>")) {
    return Family::ChatML;
  }
  if (tokenizer.token_id("<|start_header_id|>")) {
    return Family::Llama3;
  }
  if (tokenizer.token_id("<|user|>") ||
      tokenizer.chat_template().find("<|user|>") != std::string_view::npos) {
    return Family::Zephyr;
  }
  return Family::Plain;
}

std::string capitalised(std::string_view role) {
  std::string out(role);
  if (!out.empty()) {
    out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  }
  return out;
}

}  // namespace

ChatTemplate::ChatTemplate(std::optional<Template> tmpl, Family fallback, std::string bos,
                           std::string eos)
    : template_(std::move(tmpl)), fallback_(fallback), bos_(std::move(bos)), eos_(std::move(eos)) {}

ChatTemplate ChatTemplate::from_source(std::string_view source, Family fallback, std::string bos,
                                       std::string eos) {
  std::optional<Template> tmpl;
  if (!source.empty()) {
    auto parsed = Template::parse(source);
    if (parsed) {
      tmpl.emplace(std::move(*parsed));
    } else {
      log::warn("chat template: {}. Falling back to the {} format.", parsed.error().message,
                family_name(fallback));
    }
  }
  return ChatTemplate(std::move(tmpl), fallback, std::move(bos), std::move(eos));
}

ChatTemplate ChatTemplate::from_tokenizer(const Tokenizer& tokenizer) {
  const Family family = detect_family(tokenizer);
  const std::string bos(tokenizer.token_text(tokenizer.bos()));
  const std::string eos(tokenizer.token_text(tokenizer.eos()));
  return from_source(tokenizer.chat_template(), family, bos, eos);
}

std::string_view ChatTemplate::source() const {
  return template_ ? std::string_view("jinja") : family_name(fallback_);
}

Expected<std::string> ChatTemplate::render_jinja(std::span<const Message> messages,
                                                 bool add_generation_prompt,
                                                 const RenderOptions& opts) const {
  if (!template_) {
    return make_error(ErrorCode::Unsupported, "chat template: no Jinja template available");
  }
  List items;
  items.reserve(messages.size());
  for (const auto& m : messages) {
    items.push_back(
        Value::map({{"role", Value::string(m.role)}, {"content", Value::string(m.content)}}));
  }
  Map vars = {{"messages", Value::list(std::move(items))},
              {"add_generation_prompt", Value::boolean(add_generation_prompt)},
              {"bos_token", Value::string(bos_)},
              {"eos_token", Value::string(eos_)}};
  if (!opts.date_string.empty()) {
    vars.emplace_back("date_string", Value::string(opts.date_string));
  }
  return template_->render(vars);
}

std::string ChatTemplate::render_fallback(std::span<const Message> messages,
                                          bool add_generation_prompt) const {
  std::string out;
  switch (fallback_) {
  case Family::ChatML:
    for (const auto& m : messages) {
      out += fmt::format("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
    }
    if (add_generation_prompt) {
      out += "<|im_start|>assistant\n";
    }
    return out;
  case Family::Llama3:
    out += bos_;
    for (const auto& m : messages) {
      out +=
          fmt::format("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", m.role, m.content);
    }
    if (add_generation_prompt) {
      out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return out;
  case Family::Zephyr:
    for (const auto& m : messages) {
      out += fmt::format("<|{}|>\n{}{}\n", m.role, m.content, eos_);
    }
    if (add_generation_prompt) {
      out += "<|assistant|>\n";
    }
    return out;
  case Family::Plain:
    for (const auto& m : messages) {
      out += fmt::format("{}: {}\n", capitalised(m.role), m.content);
    }
    if (add_generation_prompt) {
      out += "Assistant:";
    }
    return out;
  }
  return out;
}

std::string ChatTemplate::render(std::span<const Message> messages, bool add_generation_prompt,
                                 const RenderOptions& opts) const {
  if (template_) {
    auto rendered = render_jinja(messages, add_generation_prompt, opts);
    if (rendered) {
      return *rendered;
    }
    log::warn("chat template: {}. Falling back to the {} format for this render.",
              rendered.error().message, family_name(fallback_));
  }
  return render_fallback(messages, add_generation_prompt);
}

}  // namespace axon::chat
