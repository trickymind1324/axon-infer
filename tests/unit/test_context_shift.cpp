// The context shift: a generation that fills the cache keeps the first `keep` tokens, drops
// the oldest half of the rest, re-prefills what remains and carries on. The continuation after
// the shift equals a plain generation whose prompt is exactly the retained tokens, because
// the retained cache was rebuilt from scratch and nothing was approximated.

#include "engine/engine.h"
#include "engine/speculate.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel> load(std::size_t max_ctx) {
  auto model =
      axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l-q8_0.gguf"),
                                    {.max_seq = 8, .max_ctx = max_ctx, .n_threads = 2});
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

std::vector<std::int32_t> run(axon::engine::Engine& engine, const std::vector<std::int32_t>& prompt,
                              const axon::engine::GenerateOptions& opts,
                              axon::engine::Speculator* spec = nullptr) {
  const auto out = engine.generate(prompt, opts, nullptr, spec);
  REQUIRE(out);
  return *out;
}

}  // namespace

TEST_CASE("a shifted generation continues as a plain one from the retained tokens",
          "[engine][shift]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model = load(32);
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 10);
  const std::size_t keep = 4;

  // Without the shift the context ends the generation.
  axon::engine::Engine plain(*model);
  const auto cut = run(plain, prompt, greedy(40));
  CHECK(plain.stats().context_full);
  CHECK(cut.size() < 40);

  // With it, every token comes out and the cache was shifted once.
  auto opts = greedy(33);
  opts.context_shift = true;
  opts.keep = keep;
  axon::engine::Engine shifting(*model);
  const auto all = run(shifting, prompt, opts);
  REQUIRE(all.size() == 33);
  const auto& st = shifting.stats();
  CHECK(st.context_shifts == 1);
  CHECK_FALSE(st.context_full);
  const std::size_t g = st.last_shift_at;  // generated tokens when the shift happened
  REQUIRE(g > 0);
  REQUIRE(g < 33);
  // Up to the shift the tokens are the plain run's.
  for (std::size_t i = 0; i < g; ++i) {
    CHECK(all[i] == cut[i]);
  }
  // The kept prefix is still the front of the history.
  for (std::size_t i = 0; i < keep; ++i) {
    CHECK(shifting.history()[i] == prompt[i]);
  }

  // The retained tokens by the rule: history at the shift was the prompt plus the generated
  // tokens before the pending one, the oldest half after `keep` dropped.
  std::vector<std::int32_t> history(prompt);
  history.insert(history.end(), all.begin(), all.begin() + static_cast<std::ptrdiff_t>(g - 1));
  const std::size_t drop = (history.size() - keep) / 2;
  history.erase(history.begin() + static_cast<std::ptrdiff_t>(keep),
                history.begin() + static_cast<std::ptrdiff_t>(keep + drop));
  history.push_back(all[g - 1]);  // the pending token, fed right after the shift
  auto roomy = load(64);
  axon::engine::Engine reference(*roomy);
  const auto rest = run(reference, history, greedy(static_cast<int>(33 - g)));
  const std::vector<std::int32_t> after(all.begin() + static_cast<std::ptrdiff_t>(g), all.end());
  CHECK(rest == after);
}

TEST_CASE("a shift that cannot drop anything ends with length, and the speculator follows a shift",
          "[engine][shift]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model = load(16);
  const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 6);
  // keep covers everything: nothing to drop, so the generation ends as without the shift.
  auto opts = greedy(40);
  opts.context_shift = true;
  opts.keep = 100;
  axon::engine::Engine engine(*model);
  const auto out = run(engine, prompt, opts);
  CHECK(engine.stats().context_full);
  CHECK(engine.stats().context_shifts == 0);
  CHECK(out.size() < 40);

  // Speculative decoding across a shift gives the same tokens as plain decoding across it.
  auto draft = load(16);
  axon::engine::Speculator spec(*draft, *model, {.draft_tokens = 3});
  auto shifting = greedy(30);
  shifting.context_shift = true;
  shifting.keep = 2;
  axon::engine::Engine a(*model);
  axon::engine::Engine b(*model);
  const auto plain = run(a, prompt, shifting);
  const auto speculative = run(b, prompt, shifting, &spec);
  CHECK(a.stats().context_shifts >= 1);
  CHECK(plain.size() == 30);
  CHECK(speculative == plain);
}
