#include "engine/engine.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

constexpr std::size_t kVocab = 512;

std::unique_ptr<axon::model::LlamaModel> load_synthetic(std::size_t max_seq = 8,
                                                        std::size_t max_ctx = 64) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = max_seq, .max_ctx = max_ctx});
  if (!model) {
    FAIL("cannot load the synthetic model: " << model.error().message);
  }
  return std::move(*model);
}

axon::engine::GenerateOptions greedy(int max_tokens) {
  axon::engine::GenerateOptions opts;
  opts.max_tokens = max_tokens;
  opts.sampler.temperature = 0.0F;
  opts.sampler.repeat_penalty = 1.0F;
  return opts;
}

std::vector<std::int32_t> run(axon::engine::Engine& engine, std::span<const std::int32_t> prompt,
                              const axon::engine::GenerateOptions& opts,
                              std::string* text = nullptr) {
  auto out = engine.generate(prompt, opts, [&](std::string_view piece) {
    if (text != nullptr) {
      text->append(piece);
    }
  });
  if (!out) {
    FAIL("generate: " << out.error().message);
  }
  return *out;
}

// Greedy continuation through the cache-free forward, the cache-free reference loop.
std::vector<std::int32_t> reference_greedy(axon::model::LlamaModel& model,
                                           std::vector<std::int32_t> tokens, int n) {
  std::vector<std::int32_t> out;
  for (int i = 0; i < n; ++i) {
    const auto logits = model.forward(tokens);
    REQUIRE(logits);
    const float* last = logits->data() + (tokens.size() - 1) * kVocab;
    const auto next = static_cast<std::int32_t>(std::max_element(last, last + kVocab) - last);
    out.push_back(next);
    tokens.push_back(next);
  }
  return out;
}

}  // namespace

TEST_CASE("greedy generation through the engine matches the cache-free loop", "[engine]") {
  auto model = load_synthetic(32, 64);  // the reference loop needs whole sequences per call
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  const auto expected = reference_greedy(*model, prompt, 12);

  axon::engine::Engine engine(*model);
  auto opts = greedy(12);
  opts.stop_on_tokens = false;  // the run must always reach 12
  std::string text;
  const auto got = run(engine, prompt, opts, &text);
  CHECK(got == expected);
  CHECK(engine.stats().prompt_tokens == 10);
  CHECK(engine.stats().reused_tokens == 0);
  CHECK(engine.stats().generated_tokens == 12);
  CHECK(engine.stats().cache_positions == 10 + 12 - 1);  // the last token is never fed
  CHECK(engine.stats().seed != 0);
  CHECK_FALSE(engine.stats().context_full);
  CHECK(engine.history().size() == 21);
  CHECK_FALSE(text.empty());
}

TEST_CASE("a prompt that extends the history reuses the cached prefix", "[engine]") {
  auto model = load_synthetic(8, 64);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  axon::engine::Engine engine(*model);
  const std::vector<std::int32_t> first(tokens.begin(), tokens.begin() + 10);
  const auto reply = run(engine, first, greedy(5));
  REQUIRE(engine.stats().generated_tokens == 5);

  // Second turn: the whole first exchange plus four new tokens, as a chat template would send.
  std::vector<std::int32_t> second = first;
  second.insert(second.end(), reply.begin(), reply.end());
  second.insert(second.end(), tokens.begin() + 10, tokens.begin() + 14);
  const auto history_before =
      std::vector<std::int32_t>(engine.history().begin(), engine.history().end());
  run(engine, second, greedy(3));
  CHECK(engine.stats().prompt_tokens == second.size());
  CHECK(engine.stats().reused_tokens == history_before.size());
  CHECK(engine.stats().reused_tokens == 10 + 5 - 1);

  // The reused run must equal a run from scratch.
  axon::engine::Engine fresh(*model);
  const auto from_scratch = run(fresh, second, greedy(3));
  axon::engine::Engine again(*model);
  run(again, first, greedy(5));
  CHECK(run(again, second, greedy(3)) == from_scratch);
}

TEST_CASE("a prompt that diverges truncates to the common prefix", "[engine]") {
  auto model = load_synthetic(8, 64);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  axon::engine::Engine engine(*model);
  const std::vector<std::int32_t> first(tokens.begin(), tokens.begin() + 12);
  run(engine, first, greedy(4));
  std::vector<std::int32_t> other = first;
  other[6] = static_cast<std::int32_t>((other[6] + 1) % static_cast<std::int32_t>(kVocab));
  run(engine, other, greedy(2));
  CHECK(engine.stats().reused_tokens == 6);
  axon::engine::Engine fresh(*model);
  CHECK(run(fresh, other, greedy(2)) == run(engine, other, greedy(2)));
  // An identical prompt reuses everything but the last token.
  run(engine, other, greedy(1));
  CHECK(engine.stats().reused_tokens == other.size() - 1);
}

TEST_CASE("stop tokens end generation and are not emitted", "[engine]") {
  auto model = load_synthetic(32, 64);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  const auto expected = reference_greedy(*model, prompt, 6);
  axon::engine::Engine engine(*model);
  auto opts = greedy(6);

  // The very first sample is a stop token: nothing is generated or emitted.
  const std::vector<std::int32_t> first{expected[0]};
  opts.stop_tokens = first;
  std::string text;
  CHECK(run(engine, prompt, opts, &text).empty());
  CHECK(engine.stats().stopped_on_token);
  CHECK(engine.stats().generated_tokens == 0);
  CHECK(text.empty());

  // A stop token that never comes up leaves the run untouched.
  std::int32_t absent = 0;
  while (std::ranges::find(expected, absent) != expected.end()) {
    ++absent;
  }
  const std::vector<std::int32_t> never{absent};
  opts.stop_tokens = never;
  CHECK(run(engine, prompt, opts) == expected);
  CHECK_FALSE(engine.stats().stopped_on_token);
}

TEST_CASE("a full context stops generation and is reported", "[engine]") {
  auto model = load_synthetic(4, 16);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  axon::engine::Engine engine(*model);
  auto opts = greedy(50);
  opts.stop_on_tokens = false;
  const auto got = run(engine, prompt, opts);
  CHECK(engine.stats().context_full);
  CHECK(got.size() == 16 - 10 + 1);  // one token per cache slot, plus the one never fed
  CHECK(engine.stats().cache_positions == 16);

  const std::vector<std::int32_t> too_long(tokens.begin(), tokens.begin() + 16);
  const auto err = engine.generate(too_long, opts, nullptr);
  REQUIRE_FALSE(err);
  CHECK_THAT(err.error().message, ContainsSubstring("leaves no room in a context of 16"));
}

TEST_CASE("clear forgets the cache and the history", "[engine]") {
  auto model = load_synthetic(8, 64);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  axon::engine::Engine engine(*model);
  run(engine, prompt, greedy(3));
  CHECK_FALSE(engine.history().empty());
  engine.clear();
  CHECK(engine.history().empty());
  run(engine, prompt, greedy(3));
  CHECK(engine.stats().reused_tokens == 0);
}

TEST_CASE("seeded sampling is reproducible through the engine", "[engine]") {
  auto model = load_synthetic(8, 64);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  // The synthetic model is sharply peaked, so a hot temperature is needed for the draws to
  // vary at all.
  axon::engine::GenerateOptions opts;
  opts.max_tokens = 10;
  opts.sampler.temperature = 100.0F;
  opts.sampler.top_k = 0;
  opts.sampler.top_p = 1.0F;
  opts.sampler.min_p = 0.0F;
  opts.sampler.seed = 1234;
  opts.stop_on_tokens = false;
  axon::engine::Engine a(*model);
  axon::engine::Engine b(*model);
  const auto from_a = run(a, prompt, opts);
  const auto from_b = run(b, prompt, opts);
  CHECK(from_a == from_b);
  CHECK(a.stats().seed == 1234);
  opts.sampler.seed = 4321;
  axon::engine::Engine c(*model);
  CHECK(run(c, prompt, opts) != from_a);
}

TEST_CASE("a set cancel flag ends generation after the token in flight", "[engine]") {
  auto model = load_synthetic();
  axon::engine::Engine engine(*model);
  const std::vector<std::int32_t> prompt{1, 2, 3, 4};
  std::atomic<bool> cancel{false};
  auto opts = greedy(12);
  opts.stop_on_tokens = false;
  opts.cancel = &cancel;
  std::size_t pieces = 0;
  const auto out = engine.generate(prompt, opts, [&](std::string_view) {
    if (++pieces == 3) {
      cancel.store(true);
    }
  });
  REQUIRE(out);
  // The third piece set the flag, so the loop stops before feeding another token: at most one
  // more token than the pieces seen, since a token may decode to no text yet.
  CHECK(out->size() >= 3);
  CHECK(out->size() <= pieces + 1);
  CHECK(engine.stats().cancelled);
  CHECK_FALSE(engine.stats().context_full);

  // Without the flag the same run goes to max_tokens.
  engine.clear();
  opts.cancel = nullptr;
  const auto full = run(engine, prompt, opts);
  CHECK(full.size() == 12);
  CHECK_FALSE(engine.stats().cancelled);
}
