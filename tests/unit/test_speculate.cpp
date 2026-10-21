// Speculative decoding against the target decoding alone. With greedy sampling the output is
// the same token for token, because the target's multi-row scoring pass is bit-identical to
// its single-row decode. The synthetic files serve as draft and target pairs, which gives
// different distributions over one vocabulary. Then the rules inside an accepted run, the
// context bound, the vocabulary check, and the acceptance rule's distribution.

#include "engine/engine.h"
#include "engine/speculate.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel> load(const char* file, std::size_t max_seq = 8,
                                              std::size_t max_ctx = 64) {
  auto model =
      axon::model::LlamaModel::load(axon::test::fixture_path(std::string("gguf/") + file),
                                    {.max_seq = max_seq, .max_ctx = max_ctx, .n_threads = 2});
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
  std::vector<std::int32_t> tokens;
  std::string text;
  axon::engine::Stats stats;
};

Outcome run(axon::model::LlamaModel& target, const std::vector<std::int32_t>& prompt,
            const axon::engine::GenerateOptions& opts, axon::engine::Speculator* spec) {
  axon::engine::Engine engine(target);
  Outcome out;
  const auto tokens =
      engine.generate(prompt, opts, [&](std::string_view p) { out.text += p; }, spec);
  REQUIRE(tokens);
  out.tokens = *tokens;
  out.stats = engine.stats();
  return out;
}

}  // namespace

TEST_CASE("greedy speculation equals greedy target decoding token for token",
          "[engine][speculate]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto draft = load("synthetic-2l-q4_0.gguf");
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf"}) {
    auto target = load(file);
    REQUIRE(axon::engine::Speculator::compatible(*draft, *target));
    for (const std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{7}}) {
      axon::engine::Speculator spec(*draft, *target, {.draft_tokens = k});
      for (std::size_t len : {std::size_t{1}, std::size_t{3}, std::size_t{9}}) {
        const std::vector<std::int32_t> prompt(tokens.begin(),
                                               tokens.begin() + static_cast<std::ptrdiff_t>(len));
        const Outcome alone = run(*target, prompt, greedy(24), nullptr);
        const Outcome together = run(*target, prompt, greedy(24), &spec);
        INFO(file << " draft tokens " << k << " prompt " << len);
        CHECK(together.tokens == alone.tokens);
        CHECK(together.text == alone.text);
        CHECK(together.stats.generated_tokens == 24);
      }
      const auto& st = spec.stats();
      INFO(file << " draft tokens " << k << " acceptance " << st.acceptance() << " tokens per pass "
                << st.tokens_per_pass());
      CHECK(st.rounds > 0);
      CHECK(st.tokens_per_pass() >= 1.0);
      CHECK(st.tokens_per_pass() <= static_cast<double>(k) + 1.0);
    }
  }
}

TEST_CASE("the end rules apply inside an accepted run as in a plain decode",
          "[engine][speculate]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto draft = load("synthetic-2l-q8_0.gguf");
  auto target = load("synthetic-2l-q8_0.gguf");  // the same file: every draft is accepted
  axon::engine::Speculator spec(*draft, *target, {.draft_tokens = 4});
  const std::vector<std::int32_t> prompt{tokens[0], tokens[1], tokens[2]};

  // max_tokens inside a run.
  for (int max : {1, 2, 5, 6}) {
    const Outcome alone = run(*target, prompt, greedy(max), nullptr);
    const Outcome together = run(*target, prompt, greedy(max), &spec);
    CHECK(together.tokens == alone.tokens);
    CHECK(together.tokens.size() == static_cast<std::size_t>(max));
  }

  // A stop token in the middle of what the target says ends both at the same place.
  const Outcome reference = run(*target, prompt, greedy(12), nullptr);
  REQUIRE(reference.tokens.size() == 12);
  const std::int32_t stop = reference.tokens[5];
  auto stopping = greedy(12);
  stopping.stop_on_tokens = true;
  const std::vector<std::int32_t> stops{stop};
  stopping.stop_tokens = stops;
  const Outcome alone = run(*target, prompt, stopping, nullptr);
  const Outcome together = run(*target, prompt, stopping, &spec);
  CHECK(together.tokens == alone.tokens);
  CHECK(together.stats.stopped_on_token);
  CHECK(alone.tokens.size() <= 5);

  // The cancel flag ends the run at the next accepted token.
  std::atomic<bool> cancel{false};
  auto cancelled = greedy(40);
  cancelled.cancel = &cancel;
  std::size_t pieces = 0;
  axon::engine::Engine engine(*target);
  const auto out = engine.generate(
      prompt, cancelled,
      [&](std::string_view) {
        if (++pieces == 3) {
          cancel.store(true);
        }
      },
      &spec);
  REQUIRE(out);
  CHECK(engine.stats().cancelled);
  CHECK(out->size() < 40);

  // Near the end of the context the draft shortens and the context bound still holds.
  auto small = load("synthetic-2l-q8_0.gguf", 8, 16);
  auto small_draft = load("synthetic-2l-q8_0.gguf", 8, 16);
  axon::engine::Speculator tight(*small_draft, *small, {.draft_tokens = 4});
  const std::vector<std::int32_t> long_prompt(tokens.begin(), tokens.begin() + 10);
  const Outcome bounded_alone = run(*small, long_prompt, greedy(40), nullptr);
  const Outcome bounded = run(*small, long_prompt, greedy(40), &tight);
  CHECK(bounded.tokens == bounded_alone.tokens);
  CHECK(bounded.stats.context_full);
}

TEST_CASE("a draft with another vocabulary is refused", "[engine][speculate]") {
  auto target = load("synthetic-2l.gguf");
  auto draft = load("synthetic-2l-q4_0.gguf", 4, 32);  // shorter context than the target
  CHECK_FALSE(axon::engine::Speculator::compatible(*draft, *target));
  auto fine = load("synthetic-2l-q4_0.gguf");
  CHECK(axon::engine::Speculator::compatible(*fine, *target));
}

TEST_CASE("the acceptance rule emits the target's distribution", "[engine][speculate]") {
  // Two distributions over a small vocabulary, the drafted token drawn from q, the emitted
  // token through accept_or_resample. Its empirical distribution is held to p by total
  // variation distance, against the distance plain draws from p reach with the same count.
  constexpr std::int32_t kVocab = 16;
  const std::vector<float> p{0.30F, 0.20F, 0.15F, 0.10F,  0.08F,  0.05F,  0.04F,   0.03F,
                             0.02F, 0.01F, 0.01F, 0.005F, 0.003F, 0.001F, 0.0005F, 0.0005F};
  std::vector<float> q{0.05F, 0.05F, 0.30F, 0.10F, 0.10F, 0.10F, 0.10F,  0.05F,
                       0.05F, 0.03F, 0.02F, 0.02F, 0.01F, 0.01F, 0.005F, 0.005F};
  axon::Sampler sampler({.seed = 99}, kVocab);
  axon::Sampler plain({.seed = 7}, kVocab);
  const int n = 40000;
  std::vector<double> emitted(kVocab, 0.0);
  std::vector<double> direct(kVocab, 0.0);
  std::vector<float> scratch(kVocab);
  std::size_t accepted_count = 0;
  for (int i = 0; i < n; ++i) {
    const std::int32_t drafted = sampler.draw(q);
    bool accepted = false;
    const std::int32_t token =
        axon::engine::accept_or_resample(p, q, drafted, sampler, scratch, &accepted);
    emitted[static_cast<std::size_t>(token)] += 1.0 / n;
    accepted_count += accepted ? 1 : 0;
    direct[static_cast<std::size_t>(plain.draw(p))] += 1.0 / n;
  }
  double tv_emitted = 0.0;
  double tv_direct = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(kVocab); ++i) {
    tv_emitted += 0.5 * std::abs(emitted[i] - static_cast<double>(p[i]));
    tv_direct += 0.5 * std::abs(direct[i] - static_cast<double>(p[i]));
  }
  // Measured 2026-10-22: 0.0046 for the speculative tokens against 0.0065 for plain draws,
  // both the sampling noise of 40000 draws over 16 outcomes, with 60 percent of the drafts
  // accepted. Locked at four times the measurement.
  INFO("total variation from p: speculative " << tv_emitted << ", plain draws " << tv_direct
                                              << ", accepted " << accepted_count << " of " << n);
  CHECK(tv_emitted < 0.019);
  CHECK(tv_direct < 0.026);
  CHECK(accepted_count > 0);
  CHECK(accepted_count < static_cast<std::size_t>(n));
}
