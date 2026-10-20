// SmolLM2 behind the HTTP server: a chat completion through the model's own template answers
// the question, streaming carries the same text, and the model list names the file. Runs
// only with AXON_MODELS_DIR.

#if defined(AXON_HAVE_SERVER)

#include "model/llama.h"
#include "server/http.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

namespace {

using Catch::Matchers::ContainsSubstring;
using nlohmann::json;

std::unique_ptr<axon::model::LlamaModel> load_smollm2() {
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set; run scripts/download_models.sh and export it");
  }
  const auto path = std::filesystem::path(env) / "SmolLM2-135M-Instruct-f16.gguf";
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::model::LlamaModel::load(path, {.max_seq = 64, .max_ctx = 256});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

}  // namespace

TEST_CASE("SmolLM2 answers a chat completion over HTTP, streamed and not",
          "[integration][server]") {
  auto model = load_smollm2();
  axon::server::ServerOptions options;
  options.port = 0;
  options.served_name = "smollm2";
  axon::server::HttpServer server(*model, options);
  REQUIRE(server.start());
  httplib::Client cli("127.0.0.1", server.port());
  cli.set_read_timeout(120, 0);

  CHECK(json::parse(cli.Get("/v1/models")->body)["data"][0]["id"] == "smollm2");

  const json request = {
      {"model", "smollm2"},
      {"messages",
       json::array({{{"role", "user"}, {"content", "What is the capital of France?"}}})},
      {"max_tokens", 24},
      {"temperature", 0}};
  auto r = cli.Post("/v1/chat/completions", request.dump(), "application/json");
  REQUIRE(r);
  REQUIRE(r->status == 200);
  const json body = json::parse(r->body);
  const std::string content = body["choices"][0]["message"]["content"];
  WARN("SmolLM2 over HTTP replied: " << content);
  CHECK_THAT(content, ContainsSubstring("Paris"));
  CHECK(body["usage"]["prompt_tokens"].get<int>() > 0);

  json streamed = request;
  streamed["stream"] = true;
  auto s = cli.Post("/v1/chat/completions", streamed.dump(), "application/json");
  REQUIRE(s);
  REQUIRE(s->status == 200);
  std::string text;
  std::string finish;
  std::size_t pos = 0;
  while ((pos = s->body.find("data: ", pos)) != std::string::npos) {
    const auto end = s->body.find("\n\n", pos);
    const std::string payload = s->body.substr(pos + 6, end - pos - 6);
    pos = end;
    if (payload == "[DONE]") {
      break;
    }
    const json chunk = json::parse(payload);
    const auto& choice = chunk["choices"][0];
    if (choice["delta"].contains("content")) {
      text += choice["delta"]["content"].get<std::string>();
    }
    if (!choice["finish_reason"].is_null()) {
      finish = choice["finish_reason"];
    }
  }
  CHECK(text == content);
  CHECK((finish == "stop" || finish == "length"));
  CHECK(server.requests() == 2);
}

TEST_CASE("two SmolLM2 conversations interleaved turn by turn each get the single-client answer",
          "[integration][server][batch]") {
  // One server with two slots. Each conversation asks two questions, resending its whole
  // transcript the second time, and the turns alternate so both are live at once. The answers
  // must equal what each conversation gets on a server of its own, and the second turn must
  // reuse the slot's cached prefix.
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set; run scripts/download_models.sh and export it");
  }
  const auto path = std::filesystem::path(env) / "SmolLM2-135M-Instruct-f16.gguf";
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::model::LlamaModel::load(path, {.max_seq = 64, .max_ctx = 512, .max_batch = 2});
  REQUIRE(model);

  const auto ask = [](httplib::Client& cli, const json& messages) {
    const json req = {{"messages", messages}, {"max_tokens", 24}, {"temperature", 0}};
    auto r = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    return json::parse(r->body)["choices"][0]["message"]["content"].get<std::string>();
  };
  const json france =
      json::array({{{"role", "user"}, {"content", "What is the capital of France?"}}});
  const json italy =
      json::array({{{"role", "user"}, {"content", "What is the capital of Italy?"}}});
  const auto follow = [](json transcript, const std::string& answer, const char* question) {
    transcript.push_back({{"role", "assistant"}, {"content", answer}});
    transcript.push_back({{"role", "user"}, {"content", question}});
    return transcript;
  };

  // Alone: each conversation on a one-slot server.
  std::string alone_a1;
  std::string alone_a2;
  std::string alone_b1;
  std::string alone_b2;
  {
    axon::server::ServerOptions one;
    one.port = 0;
    one.max_batch = 1;
    axon::server::HttpServer server(**model, one);
    REQUIRE(server.start());
    httplib::Client cli("127.0.0.1", server.port());
    cli.set_read_timeout(120, 0);
    alone_a1 = ask(cli, france);
    alone_a2 = ask(cli, follow(france, alone_a1, "And its population?"));
    alone_b1 = ask(cli, italy);
    alone_b2 = ask(cli, follow(italy, alone_b1, "And its population?"));
  }

  axon::server::ServerOptions two;
  two.port = 0;
  two.max_batch = 2;
  axon::server::HttpServer server(**model, two);
  REQUIRE(server.start());
  httplib::Client a("127.0.0.1", server.port());
  httplib::Client b("127.0.0.1", server.port());
  a.set_read_timeout(120, 0);
  b.set_read_timeout(120, 0);
  std::string got_a1;
  std::string got_b1;
  {
    std::thread ta([&] { got_a1 = ask(a, france); });
    got_b1 = ask(b, italy);
    ta.join();
  }
  std::string got_a2;
  std::string got_b2;
  {
    std::thread ta([&] { got_a2 = ask(a, follow(france, got_a1, "And its population?")); });
    got_b2 = ask(b, follow(italy, got_b1, "And its population?"));
    ta.join();
  }
  CHECK(got_a1 == alone_a1);
  CHECK(got_b1 == alone_b1);
  CHECK(got_a2 == alone_a2);
  CHECK(got_b2 == alone_b2);
  WARN("conversation a: " << got_a1 << " | " << got_a2);
  const json health = json::parse(a.Get("/health")->body);
  CHECK(health["slots"] == 2);
  CHECK(health["requests"] == 4);
  CHECK(health["batched_steps"].get<int>() > 0);
}

#endif
