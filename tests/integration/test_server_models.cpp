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

#endif
