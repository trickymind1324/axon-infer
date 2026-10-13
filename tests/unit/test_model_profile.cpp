// The per-launch profile: a forward call with a profile charges every slot a non-negative time
// that adds up to the call, the matmul slots are the ones that do the work, and the logits are
// the same with and without it.

#include "engine/bench.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/" + file),
                                             {.max_seq = 8, .max_ctx = 32, .n_threads = 2});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

}  // namespace

TEST_CASE("a forward profile charges the launches and leaves the logits alone",
          "[model][profile]") {
  using axon::model::ForwardProfile;
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model = load("synthetic-2l-q8_0.gguf");
  const auto plain = model->forward(std::span(tokens).first(8));
  REQUIRE(plain);
  const std::vector<float> expected(plain->begin(), plain->end());

  ForwardProfile profile;
  axon::model::ForwardOptions opts;
  opts.profile = &profile;
  const auto profiled = model->forward(std::span(tokens).first(8), opts);
  REQUIRE(profiled);
  CHECK(std::vector<float>(profiled->begin(), profiled->end()) == expected);
  for (const double s : profile.seconds) {
    CHECK(s >= 0.0);
  }
  CHECK(profile.total() > 0.0);
  CHECK(profile.seconds[ForwardProfile::kQkv] > 0.0);
  CHECK(profile.seconds[ForwardProfile::kAttention] > 0.0);
  CHECK(profile.seconds[ForwardProfile::kGateUp] > 0.0);
  CHECK(profile.seconds[ForwardProfile::kDown] > 0.0);
  CHECK(profile.seconds[ForwardProfile::kLogits] > 0.0);
  CHECK(profile.seconds[ForwardProfile::kQuantize] > 0.0);  // Q8_0 weights quantize the rows
  CHECK(std::string(ForwardProfile::name(ForwardProfile::kGateUp)) == "gate_up");
}

TEST_CASE("the F32 model quantizes nothing and the report covers the prompt and one token",
          "[model][profile]") {
  using axon::model::ForwardProfile;
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model = load("synthetic-2l.gguf");
  const auto report = axon::engine::profile_forward(*model, std::span(tokens).first(20));
  REQUIRE(report);
  CHECK(report->prompt_tokens == 20);
  CHECK(report->prefill.seconds[ForwardProfile::kQuantize] == 0.0);
  CHECK(report->decode.seconds[ForwardProfile::kQuantize] == 0.0);
  CHECK(report->prefill.total() > 0.0);
  CHECK(report->decode.total() > 0.0);
  const auto text = axon::engine::profile_text(*report);
  CHECK(text.find("gate_up") != std::string::npos);
  CHECK(text.find("total") != std::string::npos);

  // The prompt plus one token must fit the context.
  auto small = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = 8, .max_ctx = 16});
  REQUIRE(small);
  CHECK_FALSE(axon::engine::profile_forward(**small, std::span(tokens).first(16)));
}
