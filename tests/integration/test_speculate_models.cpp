// Speculative decoding on the real pairs: greedy speculation equals greedy target decoding
// token for token on the prompt fixtures, and the acceptance rate is reported. Runs only with
// AXON_MODELS_DIR and the two target files.

#include "core/axt.h"

#include "engine/engine.h"
#include "engine/speculate.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

std::filesystem::path models_dir() {
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set; run scripts/download_models.sh and export it");
  }
  return env;
}

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file) {
  const auto path = models_dir() / file;
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::model::LlamaModel::load(path, {.max_seq = 64, .max_ctx = 256});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

std::vector<std::int32_t> prompt_tokens(int i) {
  auto tensor = axon::read_axt(std::filesystem::path(AXON_FIXTURE_DIR) /
                               ("model/smollm2." + std::to_string(i) + ".tokens.axt"));
  if (!tensor) {
    FAIL(tensor.error().message);
  }
  return tensor->as_i32();
}

axon::engine::GenerateOptions greedy(int max_tokens) {
  axon::engine::GenerateOptions opts;
  opts.max_tokens = max_tokens;
  opts.sampler.temperature = 0.0F;
  opts.sampler.repeat_penalty = 1.0F;
  return opts;
}

struct Pair {
  const char* draft;
  const char* target;
  const char* question;
};

}  // namespace

TEST_CASE("greedy speculation equals greedy target decoding on the real pairs",
          "[integration][speculate]") {
  for (const Pair& pair : {Pair{.draft = "SmolLM2-135M-Instruct-Q4_0.gguf",
                                .target = "SmolLM2-1.7B-Instruct-Q4_0.gguf",
                                .question = "smollm2"},
                           Pair{.draft = "qwen2.5-0.5b-instruct-q4_0.gguf",
                                .target = "qwen2.5-1.5b-instruct-q4_0.gguf",
                                .question = "qwen"}}) {
    auto draft = load(pair.draft);
    auto target = load(pair.target);
    REQUIRE(axon::engine::Speculator::compatible(*draft, *target));
    axon::engine::Speculator spec(*draft, *target, {.draft_tokens = 4});
    // The SmolLM2 prompt fixtures are its own tokens. The Qwen pair gets its own prompts
    // through its tokenizer.
    std::vector<std::vector<std::int32_t>> prompts;
    if (std::string(pair.question) == "smollm2") {
      for (int i = 0; i < 4; ++i) {
        prompts.push_back(prompt_tokens(i));
      }
    } else {
      for (const char* text : {"The capital of France is", "Write a haiku about the sea.",
                               "Explain what a hash table is in two sentences.",
                               "List three primes greater than ten."}) {
        prompts.push_back(target->tokenizer().encode(text, {.parse_special = false}));
      }
    }
    for (const auto& prompt : prompts) {
      axon::engine::Engine alone(*target);
      axon::engine::Engine together(*target);
      std::string text_alone;
      std::string text_together;
      const auto a =
          alone.generate(prompt, greedy(32), [&](std::string_view p) { text_alone += p; });
      const auto b = together.generate(
          prompt, greedy(32), [&](std::string_view p) { text_together += p; }, &spec);
      REQUIRE(a);
      REQUIRE(b);
      INFO(pair.target << ": " << text_alone);
      CHECK(*b == *a);
      CHECK(text_together == text_alone);
    }
    const auto& st = spec.stats();
    WARN(pair.draft << " -> " << pair.target << ": " << st.rounds << " rounds, acceptance "
                    << 100.0 * st.acceptance() << "%, " << st.tokens_per_pass()
                    << " tokens per target pass");
    CHECK(st.acceptance() > 0.0);
  }
}
