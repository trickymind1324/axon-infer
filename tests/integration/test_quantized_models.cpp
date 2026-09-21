// Quantized SmolLM2 against its F16 sibling, perplexity against the Hugging Face reference, and
// a chat turn on each of the other downloaded models. Runs only with AXON_MODELS_DIR.

#include "core/axt.h"

#include "chat/template.h"
#include "engine/engine.h"
#include "engine/perplexity.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file,
                                              axon::model::LoadOptions options) {
  const auto path = models_dir() / file;
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::model::LlamaModel::load(path, options);
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

std::vector<float> forward(axon::model::LlamaModel& model, const std::vector<std::int32_t>& t) {
  const auto logits = model.forward(t);
  if (!logits) {
    FAIL(logits.error().message);
  }
  return {logits->begin(), logits->end()};
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    FAIL("cannot open " << path.string());
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr const char* kF16 = "SmolLM2-135M-Instruct-f16.gguf";
constexpr const char* kQ8 = "SmolLM2-135M-Instruct-Q8_0.gguf";
constexpr const char* kQ4 = "SmolLM2-135M-Instruct-Q4_0.gguf";

}  // namespace

TEST_CASE("quantized SmolLM2 agrees with the F16 model on the prompt fixtures",
          "[integration][model][quant]") {
  auto f16 = load(kF16, {.max_seq = 64, .max_ctx = 64});
  const std::size_t n_vocab = f16->config().n_vocab;
  struct Case {
    const char* file;
    float min_agreement;
    float max_abs_guard;
  };
  for (const auto& c : {Case{.file = kQ8, .min_agreement = 0.98F, .max_abs_guard = 0.1F},
                        Case{.file = kQ4, .min_agreement = 0.95F, .max_abs_guard = 1.0F}}) {
    auto quantized = load(c.file, {.max_seq = 64, .max_ctx = 64});
    std::size_t positions = 0;
    std::size_t agree = 0;
    float worst_last = 0.0F;
    for (int i = 0; i < 10; ++i) {
      const auto tokens = prompt_tokens(i);
      const auto reference = forward(*f16, tokens);
      const auto got = forward(*quantized, tokens);
      for (std::size_t p = 0; p < tokens.size(); ++p) {
        const float* a = reference.data() + p * n_vocab;
        const float* b = got.data() + p * n_vocab;
        agree += (std::max_element(a, a + n_vocab) - a) == (std::max_element(b, b + n_vocab) - b)
                     ? 1
                     : 0;
        ++positions;
      }
      const float* a = reference.data() + (tokens.size() - 1) * n_vocab;
      const float* b = got.data() + (tokens.size() - 1) * n_vocab;
      for (std::size_t j = 0; j < n_vocab; ++j) {
        worst_last = std::max(worst_last, std::abs(a[j] - b[j]));
      }
    }
    const float agreement = static_cast<float>(agree) / static_cast<float>(positions);
    WARN(c.file << ": argmax agreement " << agreement * 100.0F << "% over " << positions
                << " positions, max abs logit difference at the last position " << worst_last);
    CHECK(agreement >= c.min_agreement);
    CHECK(worst_last < c.max_abs_guard);
  }
}

TEST_CASE("perplexity matches Hugging Face on F16 and degrades within bounds when quantized",
          "[integration][model][quant][perplexity]") {
  // Loading first: a missing model file skips the test before any fixture is demanded.
  auto f16 = load(kF16, {.max_seq = 128, .max_ctx = 512});
  const auto text =
      read_text(std::filesystem::path(AXON_FIXTURE_DIR) / "perplexity/wikitext2-test-head.txt");
  const auto reference_file =
      std::filesystem::path(AXON_FIXTURE_DIR) / "perplexity/smollm2-135m-instruct-f16.chunks4.txt";
  std::ifstream ref(reference_file);
  if (!ref) {
    FAIL("missing " << reference_file.string() << ", run gen_perplexity_reference.py");
  }
  double ref_nll = 0.0;
  std::size_t ref_scored = 0;
  double ref_ppl = 0.0;
  ref >> ref_nll >> ref_scored >> ref_ppl;

  const axon::engine::PerplexityOptions opts{.chunk_size = 512, .max_chunks = 4};
  const auto score = [&](axon::model::LlamaModel& model, const char* file) {
    const auto tokens =
        model.tokenizer().encode(text, {.add_bos = std::nullopt, .parse_special = false});
    const auto result = axon::engine::perplexity(model, tokens, opts);
    REQUIRE(result);
    WARN(file << ": perplexity " << result->perplexity() << " over " << result->scored_tokens
              << " tokens in " << result->chunks << " chunks");
    return *result;
  };
  const auto base = score(*f16, kF16);
  CHECK(base.scored_tokens == ref_scored);
  CHECK(std::abs(base.perplexity() - ref_ppl) < 1e-3 * ref_ppl);
  f16.reset();

  auto q8 = load(kQ8, {.max_seq = 128, .max_ctx = 512});
  CHECK(score(*q8, kQ8).perplexity() <= base.perplexity() + 0.05);
  q8.reset();
  auto q4 = load(kQ4, {.max_seq = 128, .max_ctx = 512});
  CHECK(score(*q4, kQ4).perplexity() <= base.perplexity() + 0.5);
}

TEST_CASE("the other models answer a chat question through their own templates",
          "[integration][model][quant][chat]") {
  struct Case {
    const char* file;
    const char* expect;
  };
  for (const auto& c : {Case{.file = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf", .expect = "Paris"},
                        Case{.file = "qwen2.5-0.5b-instruct-q4_0.gguf", .expect = "Paris"},
                        Case{.file = "Llama-3.2-1B-Instruct-Q4_0.gguf", .expect = "Paris"}}) {
    auto model = load(c.file, {.max_seq = 64, .max_ctx = 256});
    const auto& tokenizer = model->tokenizer();
    const auto tmpl = axon::chat::ChatTemplate::from_tokenizer(tokenizer);
    INFO(c.file << " template source " << tmpl.source());
    CHECK(tmpl.source() == "jinja");
    axon::engine::Engine engine(*model);
    axon::engine::GenerateOptions opts;
    opts.max_tokens = 24;
    opts.sampler.temperature = 0.0F;
    opts.sampler.repeat_penalty = 1.0F;
    std::vector<axon::chat::Message> messages{
        {.role = "user", .content = "What is the capital of France? Answer in one sentence."}};
    const std::string prompt = tmpl.render(messages, true);
    const bool has_bos = prompt.starts_with(tokenizer.token_text(tokenizer.bos()));
    const auto ids =
        tokenizer.encode(prompt, {.add_bos = has_bos ? std::optional<bool>(false) : std::nullopt,
                                  .parse_special = true});
    std::string reply;
    const auto out =
        engine.generate(ids, opts, [&](std::string_view piece) { reply.append(piece); });
    REQUIRE(out);
    WARN(c.file << " replied: " << reply);
    CHECK(reply.find(c.expect) != std::string::npos);

    messages.push_back({.role = "assistant", .content = reply});
    messages.push_back({.role = "user", .content = "And Italy?"});
    const auto second = tokenizer.encode(
        tmpl.render(messages, true),
        {.add_bos = has_bos ? std::optional<bool>(false) : std::nullopt, .parse_special = true});
    std::string second_reply;
    const auto again =
        engine.generate(second, opts, [&](std::string_view piece) { second_reply.append(piece); });
    REQUIRE(again);
    CHECK(engine.stats().reused_tokens > ids.size() / 2);
    WARN(c.file << " second reply: " << second_reply);
  }
}
