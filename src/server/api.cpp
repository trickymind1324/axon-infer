#include "server/api.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <utility>

namespace axon::server {

namespace {

using nlohmann::json;

ApiError invalid(std::string message, std::optional<std::string> param = std::nullopt,
                 std::string code = {}) {
  return {.status = 400,
          .type = "invalid_request_error",
          .code = std::move(code),
          .message = std::move(message),
          .param = std::move(param)};
}

// Reads an optional field, refusing the wrong type. Returns nullopt when absent or null.
template <class T>
std::variant<std::optional<T>, ApiError> field(const json& body, const char* name,
                                               bool (json::*is)() const, const char* type_name) {
  const auto it = body.find(name);
  if (it == body.end() || it->is_null()) {
    return std::optional<T>{};
  }
  if (!((*it).*is)()) {
    return invalid(fmt::format("'{}' must be {}", name, type_name), name);
  }
  return std::optional<T>(it->get<T>());
}

std::variant<std::optional<double>, ApiError> number(const json& body, const char* name) {
  return field<double>(body, name, &json::is_number, "a number");
}

std::variant<std::optional<std::int64_t>, ApiError> integer(const json& body, const char* name) {
  return field<std::int64_t>(body, name, &json::is_number_integer, "an integer");
}

std::variant<std::optional<bool>, ApiError> boolean(const json& body, const char* name) {
  return field<bool>(body, name, &json::is_boolean, "a boolean");
}

// Fields the server cannot honour: present and not their neutral value means 400 naming them.
std::optional<ApiError> refuse_unsupported(const json& body, bool chat) {
  for (const char* name : {"tools", "functions", "tool_choice", "function_call", "response_format",
                           "top_logprobs", "suffix", "logit_bias"}) {
    const auto it = body.find(name);
    if (it != body.end() && !it->is_null()) {
      return invalid(fmt::format("'{}' is not supported by this server", name), name,
                     "unsupported_field");
    }
  }
  if (const auto it = body.find("logprobs"); it != body.end() && !it->is_null()) {
    const bool off = (it->is_boolean() && !it->get<bool>()) || (it->is_number() && *it == 0);
    if (!off) {
      return invalid("'logprobs' is not supported by this server", "logprobs", "unsupported_field");
    }
  }
  if (const auto it = body.find("n"); it != body.end() && !it->is_null()) {
    if (!it->is_number_integer() || *it != 1) {
      return invalid("'n' must be 1, this server returns one choice", "n");
    }
  }
  if (!chat) {
    if (const auto it = body.find("best_of"); it != body.end() && !it->is_null()) {
      if (!it->is_number_integer() || *it != 1) {
        return invalid("'best_of' must be 1", "best_of");
      }
    }
    if (const auto it = body.find("echo"); it != body.end() && !it->is_null()) {
      if (!it->is_boolean() || it->get<bool>()) {
        return invalid("'echo' is not supported by this server", "echo", "unsupported_field");
      }
    }
  }
  return std::nullopt;
}

std::variant<std::vector<std::string>, ApiError> stop_strings(const json& body) {
  std::vector<std::string> stops;
  const auto it = body.find("stop");
  if (it == body.end() || it->is_null()) {
    return stops;
  }
  if (it->is_string()) {
    stops.push_back(it->get<std::string>());
  } else if (it->is_array()) {
    for (const auto& s : *it) {
      if (!s.is_string()) {
        return invalid("'stop' must be a string or an array of strings", "stop");
      }
      stops.push_back(s.get<std::string>());
    }
  } else {
    return invalid("'stop' must be a string or an array of strings", "stop");
  }
  if (stops.size() > 4) {
    return invalid("'stop' takes at most four strings", "stop");
  }
  for (const auto& s : stops) {
    if (s.empty()) {
      return invalid("'stop' strings must not be empty", "stop");
    }
  }
  return stops;
}

// Message content is a string, or an array of text parts which are joined.
std::variant<std::string, ApiError> message_content(const json& content, std::size_t index) {
  if (content.is_string()) {
    return content.get<std::string>();
  }
  if (content.is_array()) {
    std::string joined;
    for (const auto& part : content) {
      if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") ||
          !part["text"].is_string()) {
        return invalid(fmt::format("messages[{}].content parts must be text parts", index),
                       "messages");
      }
      joined += part["text"].get<std::string>();
    }
    return joined;
  }
  return invalid(fmt::format("messages[{}].content must be a string", index), "messages");
}

std::variant<std::vector<std::int32_t>, ApiError> chat_prompt(const json& body,
                                                              const Context& ctx) {
  const auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty()) {
    return invalid("'messages' must be a non-empty array", "messages");
  }
  std::vector<chat::Message> messages;
  std::size_t index = 0;
  for (const auto& m : *it) {
    if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
      return invalid(fmt::format("messages[{}] needs a string role", index), "messages");
    }
    const auto role = m["role"].get<std::string>();
    if (role != "system" && role != "user" && role != "assistant") {
      return invalid(
          fmt::format("messages[{}].role '{}' is not system, user or assistant", index, role),
          "messages");
    }
    if (!m.contains("content")) {
      return invalid(fmt::format("messages[{}] needs content", index), "messages");
    }
    auto content = message_content(m["content"], index);
    if (std::holds_alternative<ApiError>(content)) {
      return std::get<ApiError>(content);
    }
    messages.push_back({.role = role, .content = std::get<std::string>(std::move(content))});
    ++index;
  }
  const std::string text = ctx.chat_template->render(messages, true);
  // The same rule as the chat command: the template may already start with the BOS text.
  const auto& tok = *ctx.tokenizer;
  std::optional<bool> add_bos = std::nullopt;
  if (text.starts_with(tok.token_text(tok.bos()))) {
    add_bos = false;
  }
  return tok.encode(text, {.add_bos = add_bos, .parse_special = true});
}

std::variant<std::vector<std::int32_t>, ApiError> text_prompt(const json& body,
                                                              const Context& ctx) {
  const auto it = body.find("prompt");
  if (it == body.end() || !it->is_string() || it->get<std::string>().empty()) {
    return invalid("'prompt' must be a non-empty string", "prompt");
  }
  // Untrusted text: special tokens written literally stay plain text.
  return ctx.tokenizer->encode(it->get<std::string>(),
                               {.add_bos = std::nullopt, .parse_special = false});
}

std::string random_suffix(std::size_t length) {
  static constexpr std::string_view kAlphabet =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::random_device device;
  std::mt19937_64 rng(device());
  std::uniform_int_distribution<std::size_t> pick(0, kAlphabet.size() - 1);
  std::string out;
  out.reserve(length);
  for (std::size_t i = 0; i < length; ++i) {
    out.push_back(kAlphabet[pick(rng)]);
  }
  return out;
}

json usage_json(const Usage& usage) {
  return {{"prompt_tokens", usage.prompt_tokens},
          {"completion_tokens", usage.completion_tokens},
          {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
}

}  // namespace

std::variant<json, ApiError> parse_body(std::string_view text) {
  json body = json::parse(text, nullptr, false);
  if (body.is_discarded()) {
    return invalid("the request body is not valid JSON", std::nullopt, "invalid_json");
  }
  if (!body.is_object()) {
    return invalid("the request body must be a JSON object", std::nullopt, "invalid_json");
  }
  return body;
}

ParseResult parse_request(const json& body, const Context& ctx, bool chat) {
  CompletionRequest req;
  req.chat = chat;
  req.model = ctx.served_name;
  if (const auto it = body.find("model"); it != body.end() && !it->is_null()) {
    if (!it->is_string()) {
      return invalid("'model' must be a string", "model");
    }
    if (it->get<std::string>() != ctx.served_name) {
      return ApiError{.status = 404,
                      .type = "not_found_error",
                      .code = "model_not_found",
                      .message = fmt::format("model '{}' is not served, this server serves '{}'",
                                             it->get<std::string>(), ctx.served_name),
                      .param = "model"};
    }
  }
  if (auto refused = refuse_unsupported(body, chat)) {
    return *refused;
  }

  auto stream = boolean(body, "stream");
  if (std::holds_alternative<ApiError>(stream)) {
    return std::get<ApiError>(stream);
  }
  req.stream = std::get<std::optional<bool>>(stream).value_or(false);
  if (const auto it = body.find("stream_options"); it != body.end() && !it->is_null()) {
    if (!it->is_object()) {
      return invalid("'stream_options' must be an object", "stream_options");
    }
    auto include = boolean(*it, "include_usage");
    if (std::holds_alternative<ApiError>(include)) {
      return std::get<ApiError>(include);
    }
    req.include_usage = std::get<std::optional<bool>>(include).value_or(false);
  }

  // Sampling: the OpenAI fields with the OpenAI defaults, the extensions with the sampler's.
  req.gen.sampler.temperature = 1.0F;
  req.gen.sampler.top_p = 1.0F;
  auto temperature = number(body, "temperature");
  if (std::holds_alternative<ApiError>(temperature)) {
    return std::get<ApiError>(temperature);
  }
  if (const auto t = std::get<std::optional<double>>(temperature)) {
    if (*t < 0.0 || *t > 2.0) {
      return invalid("'temperature' must be between 0 and 2", "temperature");
    }
    req.gen.sampler.temperature = static_cast<float>(*t);
  }
  auto top_p = number(body, "top_p");
  if (std::holds_alternative<ApiError>(top_p)) {
    return std::get<ApiError>(top_p);
  }
  if (const auto p = std::get<std::optional<double>>(top_p)) {
    if (*p <= 0.0 || *p > 1.0) {
      return invalid("'top_p' must be above 0 and at most 1", "top_p");
    }
    req.gen.sampler.top_p = static_cast<float>(*p);
  }
  auto min_p = number(body, "min_p");
  if (std::holds_alternative<ApiError>(min_p)) {
    return std::get<ApiError>(min_p);
  }
  if (const auto p = std::get<std::optional<double>>(min_p)) {
    if (*p < 0.0 || *p > 1.0) {
      return invalid("'min_p' must be between 0 and 1", "min_p");
    }
    req.gen.sampler.min_p = static_cast<float>(*p);
  }
  auto top_k = integer(body, "top_k");
  if (std::holds_alternative<ApiError>(top_k)) {
    return std::get<ApiError>(top_k);
  }
  if (const auto k = std::get<std::optional<std::int64_t>>(top_k)) {
    if (*k < 0 || *k > 100000) {
      return invalid("'top_k' must be between 0 and 100000", "top_k");
    }
    req.gen.sampler.top_k = static_cast<int>(*k);
  }
  auto penalty = number(body, "repetition_penalty");
  if (std::holds_alternative<ApiError>(penalty)) {
    return std::get<ApiError>(penalty);
  }
  if (const auto r = std::get<std::optional<double>>(penalty)) {
    if (*r <= 0.0 || *r > 10.0) {
      return invalid("'repetition_penalty' must be above 0 and at most 10", "repetition_penalty");
    }
    req.gen.sampler.repeat_penalty = static_cast<float>(*r);
  }
  auto seed = integer(body, "seed");
  if (std::holds_alternative<ApiError>(seed)) {
    return std::get<ApiError>(seed);
  }
  if (const auto s = std::get<std::optional<std::int64_t>>(seed)) {
    if (*s < 0) {
      return invalid("'seed' must not be negative", "seed");
    }
    req.gen.sampler.seed = static_cast<std::uint64_t>(*s);  // zero keeps the clock seed
  }

  int max_tokens = ctx.default_max_tokens;
  for (const char* name : {"max_tokens", "max_completion_tokens"}) {
    auto value = integer(body, name);
    if (std::holds_alternative<ApiError>(value)) {
      return std::get<ApiError>(value);
    }
    if (const auto v = std::get<std::optional<std::int64_t>>(value)) {
      if (*v < 1 || *v > 1000000) {
        return invalid(fmt::format("'{}' must be at least 1", name), name);
      }
      max_tokens = static_cast<int>(*v);
    }
  }

  auto stops = stop_strings(body);
  if (std::holds_alternative<ApiError>(stops)) {
    return std::get<ApiError>(stops);
  }
  req.stop = std::get<std::vector<std::string>>(std::move(stops));

  auto prompt = chat ? chat_prompt(body, ctx) : text_prompt(body, ctx);
  if (std::holds_alternative<ApiError>(prompt)) {
    return std::get<ApiError>(prompt);
  }
  req.prompt = std::get<std::vector<std::int32_t>>(std::move(prompt));
  if (req.prompt.size() >= ctx.max_ctx) {
    return ApiError{.status = 413,
                    .type = "invalid_request_error",
                    .code = "context_length_exceeded",
                    .message = fmt::format("the prompt has {} tokens and the context holds {}, "
                                           "which leaves no room for a completion",
                                           req.prompt.size(), ctx.max_ctx),
                    .param = chat ? "messages" : "prompt"};
  }
  const auto room = static_cast<int>(ctx.max_ctx - req.prompt.size());
  req.gen.max_tokens = std::min(max_tokens, room);
  return req;
}

std::string completion_id(bool chat) {
  return (chat ? "chatcmpl-" : "cmpl-") + random_suffix(24);
}

std::int64_t now_unix() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

json error_json(const ApiError& error) {
  json e = {{"message", error.message}, {"type", error.type}};
  e["param"] = error.param ? json(*error.param) : json(nullptr);
  e["code"] = error.code.empty() ? json(nullptr) : json(error.code);
  return {{"error", e}};
}

json models_json(std::string_view served_name, std::int64_t created) {
  return {{"object", "list"},
          {"data", json::array({{{"id", served_name},
                                 {"object", "model"},
                                 {"created", created},
                                 {"owned_by", "axon-infer"}}})}};
}

json health_json(std::string_view served_name, double uptime_s, std::uint64_t requests,
                 std::size_t queue) {
  return {{"status", "ok"},
          {"model", served_name},
          {"uptime_s", uptime_s},
          {"requests", requests},
          {"queue", queue}};
}

json completion_json(const CompletionRequest& req, std::string_view id, std::int64_t created,
                     std::string_view text, std::string_view finish_reason, const Usage& usage) {
  json choice;
  if (req.chat) {
    choice = {{"index", 0},
              {"message", {{"role", "assistant"}, {"content", text}}},
              {"finish_reason", finish_reason}};
  } else {
    choice = {
        {"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", finish_reason}};
  }
  return {{"id", id},
          {"object", req.chat ? "chat.completion" : "text_completion"},
          {"created", created},
          {"model", req.model},
          {"choices", json::array({choice})},
          {"usage", usage_json(usage)}};
}

json chunk_json(const CompletionRequest& req, std::string_view id, std::int64_t created,
                std::optional<std::string_view> role, std::string_view text,
                std::optional<std::string_view> finish_reason, const std::optional<Usage>& usage) {
  json choice;
  if (req.chat) {
    json delta = json::object();
    if (role) {
      delta["role"] = *role;
    }
    if (role || !text.empty()) {
      delta["content"] = text;
    }
    choice = {{"index", 0}, {"delta", delta}};
  } else {
    choice = {{"index", 0}, {"text", text}, {"logprobs", nullptr}};
  }
  choice["finish_reason"] = finish_reason ? json(*finish_reason) : json(nullptr);
  json out = {{"id", id},
              {"object", req.chat ? "chat.completion.chunk" : "text_completion"},
              {"created", created},
              {"model", req.model},
              {"choices", json::array({choice})}};
  if (usage) {
    out["usage"] = usage_json(*usage);
  }
  return out;
}

std::string sse_event(const json& payload) {
  return "data: " + payload.dump() + "\n\n";
}

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}

std::string StopMatcher::feed(std::string_view piece) {
  if (hit_) {
    return {};
  }
  if (stops_.empty()) {
    return std::string(piece);
  }
  held_.append(piece);
  // The earliest completed stop string wins.
  std::size_t earliest = std::string::npos;
  for (const auto& stop : stops_) {
    const auto at = held_.find(stop);
    if (at != std::string::npos && at < earliest) {
      earliest = at;
    }
  }
  if (earliest != std::string::npos) {
    hit_ = true;
    std::string out = held_.substr(0, earliest);
    held_.clear();
    return out;
  }
  // Keep the longest tail that is still a proper prefix of some stop string.
  std::size_t keep = 0;
  for (const auto& stop : stops_) {
    const std::size_t longest = std::min(held_.size(), stop.size() - 1);
    for (std::size_t n = longest; n > keep; --n) {
      if (held_.compare(held_.size() - n, n, stop, 0, n) == 0) {
        keep = n;
        break;
      }
    }
  }
  std::string out = held_.substr(0, held_.size() - keep);
  held_.erase(0, held_.size() - keep);
  return out;
}

std::string StopMatcher::flush() {
  std::string out;
  if (!hit_) {
    out.swap(held_);
  }
  held_.clear();
  return out;
}

}  // namespace axon::server
