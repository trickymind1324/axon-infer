#pragma once

// The OpenAI-shaped half of the server: request bodies into the engine's terms, generation
// results into response bodies, and the stop-string matcher that sits between them. Nothing
// here touches a socket or a thread, so all of it is tested on JSON alone.

#include "axon/tokenizer.h"

#include "chat/template.h"
#include "engine/engine.h"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace axon::server {

// What parsing needs from the loaded model and the flags.
struct Context {
  const Tokenizer* tokenizer = nullptr;
  const chat::ChatTemplate* chat_template = nullptr;
  std::string served_name;
  int default_max_tokens = 512;
  std::size_t max_ctx = 0;
};

// An error in the OpenAI shape with the status it travels under.
struct ApiError {
  int status = 400;
  std::string type;  // "invalid_request_error", "authentication_error", "not_found_error", ...
  std::string code;  // a short machine-readable reason, may be empty
  std::string message;
  std::optional<std::string> param;  // the offending field, when there is one
};

struct CompletionRequest {
  bool chat = false;  // chat completions shape or text completion shape in the responses
  std::vector<std::int32_t> prompt;
  engine::GenerateOptions gen;  // max_tokens capped to the context, sampler filled in
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;
  std::string model;
};

using ParseResult = std::variant<CompletionRequest, ApiError>;

// `chat` selects the messages shape, otherwise the prompt shape. Validates every field the
// server honours and refuses, with 400 and the field's name, those it cannot.
ParseResult parse_request(const nlohmann::json& body, const Context& ctx, bool chat);

// Body text to JSON, or a 400 for malformed JSON or a body that is not an object.
std::variant<nlohmann::json, ApiError> parse_body(std::string_view text);

struct Usage {
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
};

// "chatcmpl-" or "cmpl-" followed by random characters.
std::string completion_id(bool chat);
std::int64_t now_unix();

nlohmann::json error_json(const ApiError& error);
nlohmann::json models_json(std::string_view served_name, std::int64_t created);
nlohmann::json health_json(std::string_view served_name, double uptime_s, std::uint64_t requests,
                           std::size_t queue);
// The one-body response, finish_reason "stop" or "length".
nlohmann::json completion_json(const CompletionRequest& req, std::string_view id,
                               std::int64_t created, std::string_view text,
                               std::string_view finish_reason, const Usage& usage);
// One streaming chunk. `role` is set on the first chunk, `text` on the content chunks,
// `finish_reason` on the last. `usage` is attached when the request asked for it.
nlohmann::json chunk_json(const CompletionRequest& req, std::string_view id, std::int64_t created,
                          std::optional<std::string_view> role, std::string_view text,
                          std::optional<std::string_view> finish_reason,
                          const std::optional<Usage>& usage);
std::string sse_event(const nlohmann::json& payload);
inline constexpr std::string_view kSseDone = "data: [DONE]\n\n";

// Holds back the longest suffix of the text that could still begin a stop string, releases
// it once it cannot, and ends the output at a completed stop string, which is never released.
class StopMatcher {
public:
  explicit StopMatcher(std::vector<std::string> stops);
  // Text that can be sent now. Once a stop string completes, hit() is true and the text up to
  // it has been released by this or an earlier call. Feeding after a hit returns nothing.
  std::string feed(std::string_view piece);
  // Releases what is held back when generation ends without a hit.
  std::string flush();
  [[nodiscard]] bool hit() const noexcept { return hit_; }

private:
  std::vector<std::string> stops_;
  std::string held_;
  bool hit_ = false;
};

}  // namespace axon::server
