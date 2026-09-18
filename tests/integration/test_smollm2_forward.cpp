// End-to-end check of the forward pass against Hugging Face on SmolLM2-135M. Runs only when
// AXON_MODELS_DIR holds the F16 file; the fixtures were produced by
// tests/reference/gen_model_fixtures.py --smollm2 from that same file.

#include "core/axt.h"

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

constexpr int kPrompts = 10;
constexpr int kGreedyPrompts = 5;
constexpr std::size_t kMaxSeq = 64;

std::unique_ptr<axon::model::LlamaModel> load_smollm2() {
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set; run scripts/download_models.sh and export it");
  }
  const auto path = std::filesystem::path(env) / "SmolLM2-135M-Instruct-f16.gguf";
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::model::LlamaModel::load(path, kMaxSeq);
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

axon::AxtTensor fixture(const std::string& name) {
  auto tensor =
      axon::read_axt(std::filesystem::path(AXON_FIXTURE_DIR) / ("model/" + name + ".axt"));
  if (!tensor) {
    FAIL(tensor.error().message);
  }
  return std::move(*tensor);
}

std::vector<float> forward(axon::model::LlamaModel& model,
                           const std::vector<std::int32_t>& tokens) {
  const auto logits = model.forward(tokens);
  if (!logits) {
    FAIL(logits.error().message);
  }
  return {logits->begin(), logits->end()};
}

std::int32_t argmax(const float* row, std::size_t n) {
  return static_cast<std::int32_t>(std::max_element(row, row + n) - row);
}

}  // namespace

TEST_CASE("SmolLM2 argmax matches Hugging Face at every position", "[integration][model]") {
  auto model = load_smollm2();
  const std::size_t n_vocab = model->config().n_vocab;
  for (int i = 0; i < kPrompts; ++i) {
    const auto tokens = fixture("smollm2." + std::to_string(i) + ".tokens").as_i32();
    const auto expected = fixture("smollm2." + std::to_string(i) + ".argmax").as_i32();
    REQUIRE(expected.size() == tokens.size());
    const auto logits = forward(*model, tokens);
    INFO("prompt " << i);
    for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
      INFO("position " << pos);
      CHECK(argmax(logits.data() + pos * n_vocab, n_vocab) == expected[pos]);
    }
  }
}

TEST_CASE("SmolLM2 last-position logits are within 1e-3 of Hugging Face", "[integration][model]") {
  auto model = load_smollm2();
  const std::size_t n_vocab = model->config().n_vocab;
  for (int i = 0; i < kPrompts; ++i) {
    const auto tokens = fixture("smollm2." + std::to_string(i) + ".tokens").as_i32();
    const auto expected = fixture("smollm2." + std::to_string(i) + ".last_logits").as_f32();
    REQUIRE(expected.size() == n_vocab);
    const auto logits = forward(*model, tokens);
    const float* last = logits.data() + (tokens.size() - 1) * n_vocab;
    float worst = 0.0F;
    for (std::size_t j = 0; j < n_vocab; ++j) {
      worst = std::max(worst, std::abs(last[j] - expected[j]));
    }
    INFO("prompt " << i << " max abs diff " << worst);
    CHECK(worst < 1e-3F);
  }
}

TEST_CASE("SmolLM2 greedy continuations match Hugging Face", "[integration][model]") {
  auto model = load_smollm2();
  const std::size_t n_vocab = model->config().n_vocab;
  std::ifstream greedy(std::filesystem::path(AXON_FIXTURE_DIR) / "model/smollm2.greedy.txt");
  REQUIRE(greedy);
  for (int i = 0; i < kGreedyPrompts; ++i) {
    std::string line;
    REQUIRE(std::getline(greedy, line));
    std::vector<std::int32_t> expected;
    std::istringstream in(line);
    for (std::int32_t id = 0; in >> id;) {
      expected.push_back(id);
    }
    REQUIRE(expected.size() == 20);
    auto tokens = fixture("smollm2." + std::to_string(i) + ".tokens").as_i32();
    std::vector<std::int32_t> generated;
    for (std::size_t step = 0; step < expected.size(); ++step) {
      const auto logits = forward(*model, tokens);
      const auto next = argmax(logits.data() + (tokens.size() - 1) * n_vocab, n_vocab);
      generated.push_back(next);
      tokens.push_back(next);
    }
    INFO("prompt " << i);
    CHECK(generated == expected);
  }
}

TEST_CASE("SmolLM2 full logits match locally generated Hugging Face outputs",
          "[integration][model][local]") {
  const char* local = std::getenv("AXON_LOCAL_FIXTURES");
  if (local == nullptr || *local == '\0') {
    SKIP("AXON_LOCAL_FIXTURES is not set; run gen_model_fixtures.py --smollm2 and point it at "
         "tests/reference/local");
  }
  auto model = load_smollm2();
  const std::size_t n_vocab = model->config().n_vocab;
  for (int i = 0; i < kPrompts; ++i) {
    const auto tokens = fixture("smollm2." + std::to_string(i) + ".tokens").as_i32();
    auto full = axon::read_axt(std::filesystem::path(local) /
                               ("smollm2." + std::to_string(i) + ".logits.axt"));
    if (!full) {
      FAIL(full.error().message);
    }
    const auto expected = full->as_f32();
    REQUIRE(expected.size() == tokens.size() * n_vocab);
    const auto logits = forward(*model, tokens);
    float worst = 0.0F;
    for (std::size_t j = 0; j < expected.size(); ++j) {
      worst = std::max(worst, std::abs(logits[j] - expected[j]));
    }
    INFO("prompt " << i << " max abs diff " << worst);
    CHECK(worst < 1e-3F);
  }
}
