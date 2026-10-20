// The server's JSON half on its own: request bodies into the engine's terms with every
// refusal named, response and chunk shapes, and the stop-string matcher on splits at every
// byte. The synthetic model supplies the tokenizer and the fallback template.

#if defined(AXON_HAVE_SERVER)

#include "chat/template.h"
#include "fixture_tensor.h"
#include "model/llama.h"
#include "server/api.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using axon::server::ApiError;
using axon::server::CompletionRequest;
using nlohmann::json;

struct Fixture {
  std::unique_ptr<axon::model::LlamaModel> model;
  axon::chat::ChatTemplate tmpl;
  axon::server::Context ctx;

  Fixture()
      : model(load()), tmpl(axon::chat::ChatTemplate::from_tokenizer(model->tokenizer())),
        ctx{.tokenizer = &model->tokenizer(),
            .chat_template = &tmpl,
            .served_name = "synthetic",
            .default_max_tokens = 16,
            .max_ctx = 64} {}

  static std::unique_ptr<axon::model::LlamaModel> load() {
    auto m = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                           {.max_seq = 8, .max_ctx = 64});
    if (!m) {
      FAIL(m.error().message);
    }
    return std::move(*m);
  }
};

ApiError error_of(const axon::server::ParseResult& r) {
  REQUIRE(std::holds_alternative<ApiError>(r));
  return std::get<ApiError>(r);
}

CompletionRequest request_of(const axon::server::ParseResult& r) {
  if (std::holds_alternative<ApiError>(r)) {
    FAIL(std::get<ApiError>(r).message);
  }
  return std::get<CompletionRequest>(r);
}

}  // namespace

TEST_CASE("the body parser refuses what is not a JSON object", "[server][api]") {
  auto bad = axon::server::parse_body("{not json");
  REQUIRE(std::holds_alternative<ApiError>(bad));
  CHECK(std::get<ApiError>(bad).status == 400);
  CHECK(std::get<ApiError>(bad).code == "invalid_json");
  auto list = axon::server::parse_body("[1, 2]");
  REQUIRE(std::holds_alternative<ApiError>(list));
  auto ok = axon::server::parse_body(R"({"prompt": "hi"})");
  REQUIRE(std::holds_alternative<json>(ok));
  CHECK(std::get<json>(ok)["prompt"] == "hi");
}

TEST_CASE("a chat request becomes a tokenized prompt with the sampler filled in", "[server][api]") {
  Fixture f;
  const json body = {{"model", "synthetic"},
                     {"messages", json::array({{{"role", "system"}, {"content", "Be brief."}},
                                               {{"role", "user"}, {"content", "Hello"}}})},
                     {"temperature", 0.0},
                     {"max_tokens", 5},
                     {"stop", json::array({"\n\n"})},
                     {"stream", true},
                     {"stream_options", {{"include_usage", true}}},
                     {"seed", 7},
                     {"top_k", 3},
                     {"min_p", 0.1},
                     {"repetition_penalty", 1.0}};
  const auto req = request_of(axon::server::parse_request(body, f.ctx, true));
  CHECK(req.chat);
  CHECK(req.model == "synthetic");
  CHECK_FALSE(req.prompt.empty());
  CHECK(req.gen.max_tokens == 5);
  CHECK(req.gen.sampler.temperature == 0.0F);
  CHECK(req.gen.sampler.top_p == 1.0F);
  CHECK(req.gen.sampler.top_k == 3);
  CHECK(req.gen.sampler.min_p == 0.1F);
  CHECK(req.gen.sampler.repeat_penalty == 1.0F);
  CHECK(req.gen.sampler.seed == 7);
  CHECK(req.stop == std::vector<std::string>{"\n\n"});
  CHECK(req.stream);
  CHECK(req.include_usage);

  // The same messages through the template directly give the same tokens.
  const std::vector<axon::chat::Message> messages{{.role = "system", .content = "Be brief."},
                                                  {.role = "user", .content = "Hello"}};
  const auto text = f.tmpl.render(messages, true);
  const auto expected = f.model->tokenizer().encode(text, {.parse_special = true});
  CHECK(req.prompt == expected);

  // Content as text parts joins them.
  json parts = body;
  parts["messages"][1]["content"] =
      json::array({{{"type", "text"}, {"text", "Hel"}}, {{"type", "text"}, {"text", "lo"}}});
  CHECK(request_of(axon::server::parse_request(parts, f.ctx, true)).prompt == expected);
}

TEST_CASE("a text completion request tokenizes the prompt and uses the defaults", "[server][api]") {
  Fixture f;
  const json body = {{"prompt", "Once upon a time"}};
  const auto req = request_of(axon::server::parse_request(body, f.ctx, false));
  CHECK_FALSE(req.chat);
  CHECK(req.prompt == f.model->tokenizer().encode("Once upon a time", {.parse_special = false}));
  CHECK(req.gen.max_tokens == 16);  // the server default
  CHECK(req.gen.sampler.temperature == 1.0F);
  CHECK(req.gen.sampler.top_p == 1.0F);
  CHECK(req.gen.sampler.top_k == axon::SamplerConfig{}.top_k);
  CHECK_FALSE(req.stream);
  CHECK(req.stop.empty());
}

TEST_CASE("max_tokens is capped to the room the context leaves", "[server][api]") {
  Fixture f;
  std::string long_prompt;
  for (int i = 0; i < 10; ++i) {
    long_prompt += "word ";  // byte-level tokens in the synthetic tokenizer, about 50 of them
  }
  const json body = {{"prompt", long_prompt}, {"max_tokens", 1000}};
  const auto req = request_of(axon::server::parse_request(body, f.ctx, false));
  CHECK(req.prompt.size() < 64);
  CHECK(std::cmp_equal(req.gen.max_tokens, 64 - req.prompt.size()));

  std::string too_long;
  for (int i = 0; i < 200; ++i) {
    too_long += "word ";
  }
  const auto err = error_of(axon::server::parse_request({{"prompt", too_long}}, f.ctx, false));
  CHECK(err.status == 413);
  CHECK(err.code == "context_length_exceeded");
}

TEST_CASE("every refusal names its field and status", "[server][api]") {
  Fixture f;
  const json ok = {{"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}};
  struct Case {
    const char* field;
    json value;
    int status;
  };
  for (const auto& c :
       {Case{.field = "model", .value = "other", .status = 404},
        Case{.field = "model", .value = 3, .status = 400},
        Case{.field = "n", .value = 2, .status = 400},
        Case{.field = "tools", .value = json::array({1}), .status = 400},
        Case{.field = "response_format", .value = {{"type", "json_object"}}, .status = 400},
        Case{.field = "logprobs", .value = true, .status = 400},
        Case{.field = "temperature", .value = 3.0, .status = 400},
        Case{.field = "temperature", .value = "hot", .status = 400},
        Case{.field = "top_p", .value = 0.0, .status = 400},
        Case{.field = "top_k", .value = -1, .status = 400},
        Case{.field = "seed", .value = -5, .status = 400},
        Case{.field = "max_tokens", .value = 0, .status = 400},
        Case{.field = "stop", .value = json::array({"a", "b", "c", "d", "e"}), .status = 400},
        Case{.field = "stop", .value = "", .status = 400},
        Case{.field = "stream", .value = "yes", .status = 400},
        Case{.field = "stream_options", .value = 1, .status = 400},
        Case{.field = "messages", .value = json::array(), .status = 400},
        Case{.field = "messages",
             .value = json::array({{{"role", "tool"}, {"content", "x"}}}),
             .status = 400},
        Case{.field = "messages", .value = json::array({{{"role", "user"}}}), .status = 400}}) {
    json body = ok;
    body[c.field] = c.value;
    INFO(c.field << " = " << c.value.dump());
    const auto err = error_of(axon::server::parse_request(body, f.ctx, true));
    CHECK(err.status == c.status);
    CHECK(err.param == std::string(c.field));
    CHECK_FALSE(err.message.empty());
  }
  // Neutral values pass: n of 1, logprobs false, nulls.
  json neutral = ok;
  neutral["n"] = 1;
  neutral["logprobs"] = false;
  neutral["tools"] = nullptr;
  request_of(axon::server::parse_request(neutral, f.ctx, true));
  // The text endpoint refuses a prompt array and echo.
  CHECK(error_of(axon::server::parse_request({{"prompt", json::array({"a"})}}, f.ctx, false))
            .status == 400);
  CHECK(error_of(axon::server::parse_request({{"prompt", "a"}, {"echo", true}}, f.ctx, false))
            .param == "echo");
}

TEST_CASE("responses and chunks have the OpenAI shapes", "[server][api]") {
  CompletionRequest chat;
  chat.chat = true;
  chat.model = "synthetic";
  const axon::server::Usage usage{.prompt_tokens = 3, .completion_tokens = 2};
  const json body =
      axon::server::completion_json(chat, "chatcmpl-x", 1700000000, "Hi there", "stop", usage);
  CHECK(body["object"] == "chat.completion");
  CHECK(body["choices"][0]["message"]["role"] == "assistant");
  CHECK(body["choices"][0]["message"]["content"] == "Hi there");
  CHECK(body["choices"][0]["finish_reason"] == "stop");
  CHECK(body["usage"]["total_tokens"] == 5);

  const json first =
      axon::server::chunk_json(chat, "chatcmpl-x", 1, "assistant", "", std::nullopt, std::nullopt);
  CHECK(first["object"] == "chat.completion.chunk");
  CHECK(first["choices"][0]["delta"]["role"] == "assistant");
  CHECK(first["choices"][0]["finish_reason"].is_null());
  const json middle = axon::server::chunk_json(chat, "chatcmpl-x", 1, std::nullopt, "Hi",
                                               std::nullopt, std::nullopt);
  CHECK(middle["choices"][0]["delta"]["content"] == "Hi");
  CHECK_FALSE(middle["choices"][0]["delta"].contains("role"));
  const json last =
      axon::server::chunk_json(chat, "chatcmpl-x", 1, std::nullopt, "", "length", usage);
  CHECK(last["choices"][0]["finish_reason"] == "length");
  CHECK(last["usage"]["completion_tokens"] == 2);
  CHECK(axon::server::sse_event(middle).starts_with("data: {"));
  CHECK(axon::server::sse_event(middle).ends_with("\n\n"));

  CompletionRequest text;
  text.model = "synthetic";
  const json t = axon::server::completion_json(text, "cmpl-x", 1, "abc", "length", usage);
  CHECK(t["object"] == "text_completion");
  CHECK(t["choices"][0]["text"] == "abc");
  CHECK(axon::server::chunk_json(text, "cmpl-x", 1, std::nullopt, "ab", std::nullopt,
                                 std::nullopt)["choices"][0]["text"] == "ab");

  const json err = axon::server::error_json({.status = 404,
                                             .type = "not_found_error",
                                             .code = "model_not_found",
                                             .message = "no",
                                             .param = std::nullopt});
  CHECK(err["error"]["code"] == "model_not_found");
  CHECK(err["error"]["param"].is_null());
  CHECK(axon::server::models_json("synthetic", 5)["data"][0]["id"] == "synthetic");
  const json health = axon::server::health_json("synthetic", 1.5, 3, 0, 2, 4, 7);
  CHECK(health["status"] == "ok");
  CHECK(health["running"] == 2);
  CHECK(health["slots"] == 4);
  CHECK(health["batched_steps"] == 7);
  CHECK(axon::server::completion_id(true).starts_with("chatcmpl-"));
  CHECK(axon::server::completion_id(false).size() == 5 + 24);
}

TEST_CASE("the stop matcher holds back only what could still match", "[server][api][stop]") {
  using axon::server::StopMatcher;
  // No stops: everything passes straight through.
  StopMatcher none({});
  CHECK(none.feed("abc") == "abc");
  CHECK(none.flush().empty());

  // A stop split across pieces at every byte, and text after it withheld.
  const std::string text = "The answer is 42.\nUser: next question";
  for (std::size_t split = 1; split < text.size(); ++split) {
    StopMatcher m({"\nUser:", "###"});
    std::string out = m.feed(text.substr(0, split));
    out += m.feed(text.substr(split));
    out += m.flush();
    INFO("split at " << split);
    CHECK(m.hit());
    CHECK(out == "The answer is 42.");
  }

  // A partial match that turns out not to be one is released.
  StopMatcher m({"<end>"});
  CHECK(m.feed("a <e") == "a ");
  CHECK(m.feed("xit") == "<exit");
  CHECK_FALSE(m.hit());
  CHECK(m.feed(" <en") == " ");  // the space cannot start a stop, the rest might
  CHECK(m.flush() == "<en");
  CHECK_FALSE(m.hit());

  // After a hit nothing more comes out.
  StopMatcher h({"STOP"});
  CHECK(h.feed("one STOP two") == "one ");
  CHECK(h.hit());
  CHECK(h.feed("three").empty());
  CHECK(h.flush().empty());
}

#endif
