// The worker and the HTTP routes on the synthetic model: requests in order, the queue bound,
// cancellation, stop strings and length through the worker, then every route and error over
// a real socket with the server on a free port, streaming equal to the one-body answer, two
// clients at once, a client that leaves mid-stream, the bearer token and the full queue.

#if defined(AXON_HAVE_SERVER)

#include "engine/batch.h"
#include "engine/engine.h"
#include "fixture_tensor.h"
#include "model/llama.h"
#include "server/http.h"
#include "server/worker.h"

#include <catch2/catch_test_macros.hpp>

#include <httplib.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;
using namespace std::chrono_literals;

std::unique_ptr<axon::model::LlamaModel> load(std::size_t max_batch = 1) {
  auto m = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/synthetic-2l.gguf"),
      {.max_seq = 8, .max_ctx = 64, .n_threads = 2, .max_batch = max_batch});
  if (!m) {
    FAIL(m.error().message);
  }
  return std::move(*m);
}

axon::server::CompletionRequest greedy_request(std::vector<std::int32_t> prompt, int max_tokens) {
  axon::server::CompletionRequest req;
  req.prompt = std::move(prompt);
  req.gen.max_tokens = max_tokens;
  req.gen.sampler.temperature = 0.0F;
  req.gen.sampler.repeat_penalty = 1.0F;
  req.gen.stop_on_tokens = false;  // the synthetic vocabulary has no meaningful end token
  return req;
}

// Drains a job and returns its text.
std::string drain(axon::server::Job& job) {
  std::string text;
  for (;;) {
    std::string piece;
    const auto poll = job.next(piece, 2s);
    if (poll == axon::server::Job::Poll::Piece) {
      text += piece;
    } else if (poll == axon::server::Job::Poll::Done) {
      return text;
    } else {
      FAIL("the job produced nothing for two seconds");
    }
  }
}

std::string reference_text(axon::model::LlamaModel& model, const std::vector<std::int32_t>& prompt,
                           int max_tokens) {
  axon::engine::Engine engine(model);
  const auto req = greedy_request(prompt, max_tokens);
  std::string text;
  const auto out = engine.generate(req.prompt, req.gen, [&](std::string_view p) { text += p; });
  REQUIRE(out);
  return text;
}

}  // namespace

TEST_CASE("the worker serves jobs in order with the engine's text", "[server][worker]") {
  auto model = load();
  const std::vector<std::int32_t> prompt{5, 6, 7, 8};
  const std::string expected = reference_text(*model, prompt, 12);
  REQUIRE_FALSE(expected.empty());

  axon::engine::BatchEngine engine(*model, 1);
  axon::server::Worker worker(engine, 8);
  std::vector<std::shared_ptr<axon::server::Job>> jobs;
  for (int i = 0; i < 3; ++i) {
    jobs.push_back(worker.submit(greedy_request(prompt, 12)));
    REQUIRE(jobs.back());
  }
  for (auto& job : jobs) {
    CHECK(drain(*job) == expected);
    CHECK(job->finish_reason() == "length");
    CHECK(job->usage().prompt_tokens == 4);
    CHECK(job->usage().completion_tokens == 12);
    CHECK_FALSE(job->error());
    CHECK(job->queue_wait_s() >= 0.0);
  }
  CHECK(worker.served() == 3);
  worker.stop();
  CHECK_FALSE(worker.submit(greedy_request(prompt, 1)));  // stopped workers refuse
}

TEST_CASE("the worker's queue is bounded and a cancelled job ends early", "[server][worker]") {
  auto model = load();
  axon::engine::BatchEngine engine(*model, 1);
  const std::vector<std::int32_t> prompt{1, 2, 3};
  {
    axon::server::Worker none(engine, 0);
    CHECK_FALSE(none.submit(greedy_request(prompt, 2)));
  }
  axon::server::Worker worker(engine, 1);
  std::size_t accepted = 0;
  std::vector<std::shared_ptr<axon::server::Job>> jobs;
  for (int i = 0; i < 20; ++i) {
    if (auto job = worker.submit(greedy_request(prompt, 4))) {
      ++accepted;
      jobs.push_back(std::move(job));
    }
  }
  CHECK(accepted >= 1);
  for (auto& job : jobs) {
    drain(*job);
    CHECK(job->done());
  }
  CHECK(worker.queue_depth() == 0);

  auto cancelled = worker.submit(greedy_request(prompt, 60));
  REQUIRE(cancelled);
  cancelled->cancel();
  drain(*cancelled);
  CHECK(cancelled->cancelled());
  CHECK(cancelled->usage().completion_tokens < 60);
}

TEST_CASE("a stop string ends the job with the text before it", "[server][worker]") {
  auto model = load();
  const std::vector<std::int32_t> prompt{9, 10, 11, 12, 13};
  const std::string full = reference_text(*model, prompt, 24);
  REQUIRE(full.size() > 8);
  // A stop string cut from the middle of what the model says.
  const std::string stop = full.substr(full.size() / 2, 3);
  const std::size_t at = full.find(stop);
  REQUIRE(at != std::string::npos);

  axon::engine::BatchEngine engine(*model, 1);
  axon::server::Worker worker(engine, 4);
  auto req = greedy_request(prompt, 24);
  req.stop = {stop};
  auto job = worker.submit(req);
  REQUIRE(job);
  CHECK(drain(*job) == full.substr(0, at));
  CHECK(job->finish_reason() == "stop");
  CHECK(job->usage().completion_tokens < 24);
}

namespace {

struct Served {
  std::unique_ptr<axon::model::LlamaModel> model;
  std::unique_ptr<axon::server::HttpServer> server;
  explicit Served(axon::server::ServerOptions options = {}, std::size_t max_batch = 1)
      : model(load(max_batch)) {
    options.port = 0;
    if (options.served_name.empty()) {
      options.served_name = "synthetic";
    }
    options.default_max_tokens = 6;
    server = std::make_unique<axon::server::HttpServer>(*model, options);
    REQUIRE(server->start());
  }
  httplib::Client client() const {
    httplib::Client cli("127.0.0.1", server->port());
    cli.set_read_timeout(30, 0);
    return cli;
  }
};

json body_of(const httplib::Result& r) {
  REQUIRE(r);
  return json::parse(r->body);
}

// Splits an event stream into its JSON payloads, checking the framing.
std::vector<json> events(const std::string& stream, bool* saw_done) {
  std::vector<json> out;
  std::size_t pos = 0;
  while (pos < stream.size()) {
    const auto end = stream.find("\n\n", pos);
    REQUIRE(end != std::string::npos);
    const std::string line = stream.substr(pos, end - pos);
    REQUIRE(line.starts_with("data: "));
    const std::string payload = line.substr(6);
    if (payload == "[DONE]") {
      *saw_done = true;
      REQUIRE(end + 2 == stream.size());  // nothing after the end marker
    } else {
      out.push_back(json::parse(payload));
    }
    pos = end + 2;
  }
  return out;
}

}  // namespace

TEST_CASE("the routes answer with the OpenAI shapes and errors", "[server][http]") {
  Served s;
  auto cli = s.client();

  auto health = body_of(cli.Get("/health"));
  CHECK(health["status"] == "ok");
  CHECK(health["model"] == "synthetic");
  CHECK(health["requests"] == 0);

  auto models = cli.Get("/v1/models");
  CHECK(models->status == 200);
  CHECK(body_of(models)["data"][0]["id"] == "synthetic");

  auto missing = cli.Get("/nothing/here");
  CHECK(missing->status == 404);
  CHECK(body_of(missing)["error"]["type"] == "not_found_error");

  auto bad = cli.Post("/v1/completions", "{oops", "application/json");
  CHECK(bad->status == 400);
  CHECK(body_of(bad)["error"]["code"] == "invalid_json");

  auto other = cli.Post(
      "/v1/chat/completions",
      json{{"model", "gpt-4"}, {"messages", json::array({{{"role", "user"}, {"content", "x"}}})}}
          .dump(),
      "application/json");
  CHECK(other->status == 404);
  CHECK(body_of(other)["error"]["code"] == "model_not_found");

  auto refused =
      cli.Post("/v1/completions", json{{"prompt", "abc"}, {"n", 3}}.dump(), "application/json");
  CHECK(refused->status == 400);
  CHECK(body_of(refused)["error"]["param"] == "n");

  const json request = {{"prompt", "abc"}, {"max_tokens", 5}, {"temperature", 0}};
  auto done = cli.Post("/v1/completions", request.dump(), "application/json");
  REQUIRE(done->status == 200);
  auto body = body_of(done);
  CHECK(body["object"] == "text_completion");
  CHECK(body["model"] == "synthetic");
  CHECK(body["choices"][0]["finish_reason"] == "length");
  CHECK(body["usage"]["completion_tokens"] == 5);
  CHECK(done->has_header("X-Axon-Queue-Ms"));
  const std::string text = body["choices"][0]["text"];
  CHECK_FALSE(text.empty());

  // The same prompt again gives the same text (greedy), and the request count moved.
  auto again = body_of(cli.Post("/v1/completions", request.dump(), "application/json"));
  CHECK(again["choices"][0]["text"] == text);
  CHECK(body_of(cli.Get("/health"))["requests"] == 2);

  const json chat = {{"messages", json::array({{{"role", "user"}, {"content", "abc"}}})},
                     {"temperature", 0}};  // default max tokens from the server
  auto reply = body_of(cli.Post("/v1/chat/completions", chat.dump(), "application/json"));
  CHECK(reply["object"] == "chat.completion");
  CHECK(reply["choices"][0]["message"]["role"] == "assistant");
  CHECK(reply["choices"][0]["finish_reason"] == "length");
  CHECK(reply["usage"]["completion_tokens"] == 6);
}

TEST_CASE("streaming delivers the same text as one body, chunk by chunk", "[server][http]") {
  Served s;
  auto cli = s.client();
  const json one = {{"prompt", "hello"}, {"max_tokens", 8}, {"temperature", 0}};
  const std::string expected =
      body_of(cli.Post("/v1/completions", one.dump(), "application/json"))["choices"][0]["text"];

  json streamed = one;
  streamed["stream"] = true;
  streamed["stream_options"] = {{"include_usage", true}};
  auto r = cli.Post("/v1/completions", streamed.dump(), "application/json");
  REQUIRE(r);
  CHECK(r->status == 200);
  CHECK(r->get_header_value("Content-Type").starts_with("text/event-stream"));
  bool saw_done = false;
  const auto chunks = events(r->body, &saw_done);
  CHECK(saw_done);
  REQUIRE(chunks.size() >= 2);
  std::string text;
  for (std::size_t i = 0; i + 1 < chunks.size(); ++i) {
    CHECK(chunks[i]["object"] == "text_completion");
    CHECK(chunks[i]["choices"][0]["finish_reason"].is_null());
    text += chunks[i]["choices"][0]["text"].get<std::string>();
  }
  CHECK(text == expected);
  CHECK(chunks.back()["choices"][0]["finish_reason"] == "length");
  CHECK(chunks.back()["usage"]["completion_tokens"] == 8);

  // The chat shape: a role first, content deltas, then the finish reason.
  const json chat = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
                     {"max_tokens", 4},
                     {"temperature", 0},
                     {"stream", true}};
  auto c = cli.Post("/v1/chat/completions", chat.dump(), "application/json");
  saw_done = false;
  const auto deltas = events(c->body, &saw_done);
  CHECK(saw_done);
  REQUIRE(deltas.size() >= 3);
  CHECK(deltas.front()["choices"][0]["delta"]["role"] == "assistant");
  CHECK(deltas.back()["choices"][0]["finish_reason"] == "length");
  CHECK_FALSE(deltas.back().contains("usage"));
  std::string content;
  for (const auto& d : deltas) {
    if (d["choices"][0]["delta"].contains("content")) {
      content += d["choices"][0]["delta"]["content"].get<std::string>();
    }
  }
  CHECK_FALSE(content.empty());
}

TEST_CASE("two clients at once each get the answer they would get alone", "[server][http]") {
  Served s;
  const std::vector<std::string> prompts{"alpha", "beta", "gamma", "delta"};
  std::vector<std::string> expected;
  {
    auto cli = s.client();
    for (const auto& p : prompts) {
      const json req = {{"prompt", p}, {"max_tokens", 6}, {"temperature", 0}};
      expected.push_back(body_of(
          cli.Post("/v1/completions", req.dump(), "application/json"))["choices"][0]["text"]);
    }
  }
  std::vector<std::string> failures;
  std::mutex failures_mutex;
  std::vector<std::thread> clients;
  clients.reserve(2);
  for (int c = 0; c < 2; ++c) {
    clients.emplace_back([&, c] {
      auto cli = s.client();
      for (int round = 0; round < 4; ++round) {
        const auto i = static_cast<std::size_t>((round + c) % 4);
        const json req = {{"prompt", prompts[i]}, {"max_tokens", 6}, {"temperature", 0}};
        auto r = cli.Post("/v1/completions", req.dump(), "application/json");
        if (!r || r->status != 200 || json::parse(r->body)["choices"][0]["text"] != expected[i]) {
          const std::scoped_lock lock(failures_mutex);
          failures.push_back(prompts[i]);
        }
      }
    });
  }
  for (auto& t : clients) {
    t.join();
  }
  CHECK(failures.empty());
  CHECK(body_of(s.client().Get("/health"))["requests"] == 4 + 8);
}

TEST_CASE("a client that leaves mid-stream does not take the server with it", "[server][http]") {
  Served s;
  auto cli = s.client();
  const json req = {{"prompt", "abc"}, {"max_tokens", 50}, {"temperature", 0}, {"stream", true}};
  std::size_t received = 0;
  auto r = cli.Post("/v1/completions", httplib::Headers{}, req.dump(), "application/json",
                    [&](const char*, std::size_t n) {
                      received += n;
                      return false;  // hang up after the first bytes
                    });
  CHECK(received > 0);
  // Whatever the client library reports for its own hang-up, the server is fine afterwards.
  auto after = s.client();
  CHECK(body_of(after.Get("/health"))["status"] == "ok");
  const json small = {{"prompt", "abc"}, {"max_tokens", 2}, {"temperature", 0}};
  CHECK(after.Post("/v1/completions", small.dump(), "application/json")->status == 200);
}

TEST_CASE("the bearer token guards the API routes and the full queue answers 503",
          "[server][http]") {
  {
    axon::server::ServerOptions guarded;
    guarded.api_key = "secret";
    Served s(guarded);
    auto cli = s.client();
    CHECK(cli.Get("/health")->status == 200);
    auto denied = cli.Get("/v1/models");
    CHECK(denied->status == 401);
    CHECK(body_of(denied)["error"]["type"] == "authentication_error");
    httplib::Headers auth{{"Authorization", "Bearer secret"}};
    CHECK(cli.Get("/v1/models", auth)->status == 200);
    CHECK(cli.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer wrong"}})->status ==
          401);
  }
  {
    axon::server::ServerOptions tiny;
    tiny.max_queue = 0;
    Served s(tiny);
    auto cli = s.client();
    const json req = {{"prompt", "abc"}, {"max_tokens", 2}};
    auto full = cli.Post("/v1/completions", req.dump(), "application/json");
    CHECK(full->status == 503);
    CHECK(full->get_header_value("Retry-After") == "1");
    CHECK(body_of(full)["error"]["code"] == "queue_full");
  }
}

TEST_CASE("several clients decode together and each gets the answer it gets alone",
          "[server][http][batch]") {
  axon::server::ServerOptions options;
  options.max_batch = 4;
  Served s(options, 4);
  const std::vector<std::string> prompts{"alpha", "beta", "gamma", "delta"};
  std::vector<std::string> expected;
  {
    auto cli = s.client();
    for (const auto& p : prompts) {
      const json req = {{"prompt", p}, {"max_tokens", 12}, {"temperature", 0}};
      expected.push_back(body_of(
          cli.Post("/v1/completions", req.dump(), "application/json"))["choices"][0]["text"]);
    }
  }
  std::vector<std::string> got(prompts.size());
  std::vector<int> batch_sizes(prompts.size(), 0);
  std::vector<std::thread> clients;
  clients.reserve(prompts.size());
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    clients.emplace_back([&, i] {
      auto cli = s.client();
      const json req = {{"prompt", prompts[i]}, {"max_tokens", 12}, {"temperature", 0}};
      auto r = cli.Post("/v1/completions", req.dump(), "application/json");
      if (r && r->status == 200) {
        got[i] = json::parse(r->body)["choices"][0]["text"];
        batch_sizes[i] = std::stoi(r->get_header_value("X-Axon-Batch"));
      }
    });
  }
  for (auto& t : clients) {
    t.join();
  }
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    INFO(prompts[i]);
    CHECK(got[i] == expected[i]);
    CHECK(batch_sizes[i] >= 1);
  }
  const json health = body_of(s.client().Get("/health"));
  CHECK(health["slots"] == 4);
  CHECK(health["running"] == 0);
  CHECK(health["requests"] == 8);
  // Four clients within a few milliseconds on a 2-layer model: at least one step batched them.
  CHECK(health["batched_steps"].get<int>() > 0);
}

#endif
