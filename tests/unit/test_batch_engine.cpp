// The batch engine against the single engine: every sequence gets the text, tokens and finish
// it gets alone, whatever the admission order, while the steps really batch. Slots go back to
// the client that continues its conversation, a cancelled sequence frees its slot, and the
// prompt rules fail the same way.

#include "engine/batch.h"
#include "engine/engine.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using axon::engine::BatchEngine;
using axon::engine::Finish;

std::unique_ptr<axon::model::LlamaModel> load(std::size_t max_batch,
                                              const char* file = "synthetic-2l.gguf") {
  auto model = axon::model::LlamaModel::load(
      axon::test::fixture_path(std::string("gguf/") + file),
      {.max_seq = 8, .max_ctx = 64, .n_threads = 2, .max_batch = max_batch});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

axon::engine::GenerateOptions greedy(int max_tokens) {
  axon::engine::GenerateOptions opts;
  opts.max_tokens = max_tokens;
  opts.sampler.temperature = 0.0F;
  opts.sampler.repeat_penalty = 1.0F;
  opts.stop_on_tokens = false;
  return opts;
}

struct Outcome {
  std::string text;
  std::vector<std::int32_t> tokens;
  std::optional<Finish> finish;
  std::size_t generated = 0;
};

Outcome alone(axon::model::LlamaModel& model, const std::vector<std::int32_t>& prompt,
              const axon::engine::GenerateOptions& opts) {
  axon::engine::Engine engine(model);
  Outcome out;
  const auto tokens = engine.generate(prompt, opts, [&](std::string_view p) { out.text += p; });
  REQUIRE(tokens);
  out.tokens = *tokens;
  out.generated = engine.stats().generated_tokens;
  return out;
}

// Drives the batch engine until nothing runs.
void drain(BatchEngine& batch) {
  while (batch.busy()) {
    const auto step = batch.step();
    REQUIRE(step);
  }
}

}  // namespace

TEST_CASE("batched sequences get the text they get alone, in any admission order",
          "[engine][batch]") {
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf"}) {
    auto model = load(4, file);
    INFO(file);
    const std::vector<std::vector<std::int32_t>> prompts{
        {1, 2, 3}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, {14}, {15, 16, 17, 18, 19}};
    const std::vector<int> lengths{12, 5, 9, 7};
    std::vector<Outcome> expected;
    expected.reserve(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      expected.push_back(alone(*model, prompts[i], greedy(lengths[i])));
    }

    for (const std::vector<std::size_t>& order :
         {std::vector<std::size_t>{0, 1, 2, 3}, std::vector<std::size_t>{3, 2, 1, 0},
          std::vector<std::size_t>{2, 0, 3, 1}}) {
      BatchEngine batch(*model, 4);
      std::vector<Outcome> got(prompts.size());
      for (const std::size_t i : order) {
        const auto slot = batch.admit(
            prompts[i], greedy(lengths[i]), [&, i](std::string_view p) { got[i].text += p; },
            [&, i](Finish finish, const axon::engine::Stats& stats) {
              got[i].finish = finish;
              got[i].generated = stats.generated_tokens;
            });
        REQUIRE(slot);
        REQUIRE(slot->has_value());
      }
      drain(batch);
      CHECK(batch.batched_steps() > 0);
      for (std::size_t i = 0; i < prompts.size(); ++i) {
        INFO("sequence " << i);
        CHECK(got[i].text == expected[i].text);
        CHECK(got[i].finish == Finish::Length);
        CHECK(got[i].generated == expected[i].generated);
      }
    }
  }
}

TEST_CASE("a late arrival joins the running batch and a cancelled one leaves it",
          "[engine][batch]") {
  auto model = load(3);
  const std::vector<std::int32_t> first{1, 2, 3, 4};
  const std::vector<std::int32_t> second{9, 8, 7};
  const Outcome ref_first = alone(*model, first, greedy(10));
  const Outcome ref_second = alone(*model, second, greedy(6));

  BatchEngine batch(*model, 3);
  std::string text_first;
  std::string text_second;
  std::optional<Finish> finish_first;
  std::optional<Finish> finish_second;
  REQUIRE(batch.admit(
      first, greedy(10), [&](std::string_view p) { text_first += p; },
      [&](Finish f, const axon::engine::Stats&) { finish_first = f; }));
  // A few steps alone, then the second arrives mid-way.
  for (int i = 0; i < 3; ++i) {
    REQUIRE(batch.step());
  }
  REQUIRE(batch.admit(
      second, greedy(6), [&](std::string_view p) { text_second += p; },
      [&](Finish f, const axon::engine::Stats&) { finish_second = f; }));
  drain(batch);
  CHECK(text_first == ref_first.text);
  CHECK(text_second == ref_second.text);
  CHECK(finish_first == Finish::Length);
  CHECK(finish_second == Finish::Length);
  CHECK(batch.batched_steps() > 0);

  // Cancel: the flag ends the sequence at its next step and the slot is free again.
  std::atomic<bool> cancel{false};
  auto opts = greedy(40);
  opts.cancel = &cancel;
  std::optional<Finish> finish_cancelled;
  std::size_t generated = 0;
  const auto slot = batch.admit(first, opts, {}, [&](Finish f, const axon::engine::Stats& s) {
    finish_cancelled = f;
    generated = s.generated_tokens;
  });
  REQUIRE(slot);
  REQUIRE(slot->has_value());
  while (batch.sequence(**slot).state() != axon::engine::Sequence::State::Decode) {
    REQUIRE(batch.step());
  }
  REQUIRE(batch.step());  // one accepted token
  cancel.store(true);
  REQUIRE(batch.step());  // seen at the next accept
  CHECK(finish_cancelled == Finish::Cancelled);
  CHECK(generated < 40);
  CHECK(batch.sequence(**slot).idle());
  CHECK_FALSE(batch.busy());
}

TEST_CASE("a continuing conversation gets its slot and prefix back", "[engine][batch]") {
  auto model = load(2);
  BatchEngine batch(*model, 2);
  const std::vector<std::int32_t> a{1, 2, 3, 4, 5};
  const std::vector<std::int32_t> b{20, 21, 22};
  REQUIRE(batch.admit(a, greedy(2), {}, {}));
  REQUIRE(batch.admit(b, greedy(2), {}, {}));
  CHECK_FALSE(batch.admit(a, greedy(2), {}, {})->has_value());  // both slots busy
  drain(batch);

  // Continue conversation a: the slot that holds it is chosen and the prefix reused.
  std::vector<std::int32_t> a2 = a;
  for (const std::int32_t t : batch.sequence(0).generated()) {
    a2.push_back(t);
  }
  a2.push_back(6);
  std::size_t reused = 0;
  const auto slot = batch.admit(
      a2, greedy(1), {}, [&](Finish, const axon::engine::Stats& s) { reused = s.reused_tokens; });
  REQUIRE(slot);
  REQUIRE(slot->has_value());
  CHECK(**slot == 0);
  drain(batch);
  // The cache holds the prompt and the one generated token that was fed, not the last one.
  CHECK(reused == a.size() + 1);

  // The rules of a prompt fail the same way as in the engine.
  CHECK_FALSE(batch.admit({}, greedy(1), {}, {}));
  std::vector<std::int32_t> too_long(64, 1);
  CHECK_FALSE(batch.admit(too_long, greedy(1), {}, {}));
}
