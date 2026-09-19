// The engine and the chat template on the real SmolLM2 model. Runs only with AXON_MODELS_DIR.

#include "core/axt.h"

#include "chat/template.h"
#include "engine/engine.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

std::vector<std::int32_t> read_ids(const std::filesystem::path& path) {
  auto tensor = axon::read_axt(path);
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

}  // namespace

TEST_CASE("SmolLM2 greedy generation through the engine matches Hugging Face",
          "[integration][model][engine]") {
  auto model = load_smollm2();
  axon::engine::Engine engine(*model);
  std::ifstream greedy_file(std::filesystem::path(AXON_FIXTURE_DIR) / "model/smollm2.greedy.txt");
  REQUIRE(greedy_file);
  for (int i = 0; i < 5; ++i) {
    std::string line;
    REQUIRE(std::getline(greedy_file, line));
    std::vector<std::int32_t> expected;
    std::istringstream in(line);
    for (std::int32_t id = 0; in >> id;) {
      expected.push_back(id);
    }
    const auto prompt = read_ids(std::filesystem::path(AXON_FIXTURE_DIR) /
                                 ("model/smollm2." + std::to_string(i) + ".tokens.axt"));
    auto opts = greedy(20);
    const std::vector<std::int32_t> none;
    opts.stop_tokens = std::span<const std::int32_t>(none).first(0);  // HF ran with min_new_tokens
    const auto got = engine.generate(prompt, opts, nullptr);
    REQUIRE(got);
    INFO("prompt " << i);
    CHECK(*got == expected);
  }
}

TEST_CASE("SmolLM2 holds a two-turn chat and reuses the cache on the second turn",
          "[integration][model][engine]") {
  auto model = load_smollm2();
  const auto& tokenizer = model->tokenizer();
  const auto tmpl = axon::chat::ChatTemplate::from_tokenizer(tokenizer);
  REQUIRE(tmpl.source() == "jinja");
  axon::engine::Engine engine(*model);
  std::vector<axon::chat::Message> messages{
      {.role = "user", .content = "What is the capital of France?"}};

  const auto encode = [&](const std::string& text) {
    return tokenizer.encode(text, {.add_bos = std::nullopt, .parse_special = true});
  };
  const auto first_ids = encode(tmpl.render(messages, true));
  std::string reply;
  const auto first =
      engine.generate(first_ids, greedy(48), [&](std::string_view piece) { reply.append(piece); });
  REQUIRE(first);
  CHECK_FALSE(first->empty());
  CHECK(engine.stats().stopped_on_token);  // the model closed its turn with <|im_end|>
  CHECK(reply.find("Paris") != std::string::npos);
  const auto first_positions = engine.stats().cache_positions;

  messages.push_back({.role = "assistant", .content = reply});
  messages.push_back({.role = "user", .content = "And of Germany?"});
  const auto second_ids = encode(tmpl.render(messages, true));
  std::string second_reply;
  const auto second = engine.generate(second_ids, greedy(48),
                                      [&](std::string_view piece) { second_reply.append(piece); });
  REQUIRE(second);
  CHECK_FALSE(second->empty());
  // The whole first exchange came from the cache. The template closes the assistant turn
  // with <|im_end|>, which the engine never fed, so the reuse falls just short of the
  // positions the first turn left behind.
  CHECK(engine.stats().reused_tokens >= first_positions - 1);
  CHECK(engine.stats().reused_tokens < second_ids.size());
  CHECK(second_reply.find("Berlin") != std::string::npos);
}
