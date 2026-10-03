// Quantized synthetic models: they load, run on both activation paths, stay close to the F32
// model, and keep every property the F32 path has (cache bit-identity, sized arena, no heap
// allocation in decode).

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kVocab = 512;

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file, bool exact = false,
                                              std::size_t max_seq = 32, std::size_t max_ctx = 32) {
  auto model = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/" + file),
      {.max_seq = max_seq, .max_ctx = max_ctx, .exact_activations = exact});
  if (!model) {
    FAIL("cannot load " << file << ": " << model.error().message);
  }
  return std::move(*model);
}

std::vector<float> forward(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens) {
  const auto logits = model.forward(tokens);
  if (!logits) {
    FAIL("forward: " << logits.error().message);
  }
  return {logits->begin(), logits->end()};
}

float argmax_agreement(const std::vector<float>& a, const std::vector<float>& b) {
  const std::size_t rows = a.size() / kVocab;
  std::size_t same = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const auto* ra = a.data() + r * kVocab;
    const auto* rb = b.data() + r * kVocab;
    const auto ia = std::max_element(ra, ra + kVocab) - ra;
    const auto ib = std::max_element(rb, rb + kVocab) - rb;
    same += ia == ib ? 1 : 0;
  }
  return static_cast<float>(same) / static_cast<float>(rows);
}

}  // namespace

TEST_CASE("quantized synthetic models stay close to the F32 model", "[model][quant]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto reference = load("synthetic-2l.gguf");
  const auto expected = forward(*reference, tokens);
  float range = 0.0F;
  for (const float v : expected) {
    range = std::max(range, std::abs(v));
  }
  struct Case {
    const char* file;
    float bound;  // fraction of the logit range
  };
  for (const auto& c : {Case{.file = "synthetic-2l-q8_0.gguf", .bound = 0.02F},
                        Case{.file = "synthetic-2l-q4_0.gguf", .bound = 0.15F},
                        Case{.file = "synthetic-2l-q4_1.gguf", .bound = 0.15F}}) {
    for (const bool exact : {false, true}) {
      auto model = load(c.file, exact);
      const auto got = forward(*model, tokens);
      const float diff = axon::test::max_abs_diff(got, expected);
      INFO(c.file << (exact ? " exact activations" : " q8 activations") << ": max abs diff " << diff
                  << " over a logit range of " << range);
      CHECK(diff < c.bound * range);
      CHECK(argmax_agreement(got, expected) >= 0.95F);
    }
  }
}

TEST_CASE("the two activation paths are each deterministic and differ only by noise",
          "[model][quant]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto q8 = load("synthetic-2l-q4_0.gguf", false);
  auto exact = load("synthetic-2l-q4_0.gguf", true);
  const auto a = forward(*q8, tokens);
  const auto b = forward(*q8, tokens);
  CHECK(a == b);
  const auto e = forward(*exact, tokens);
  CHECK(e == forward(*exact, tokens));
  CHECK(a != e);
  float range = 0.0F;
  for (const float v : e) {
    range = std::max(range, std::abs(v));
  }
  CHECK(axon::test::max_abs_diff(a, e) < 0.05F * range);
}

TEST_CASE("cached decoding on a quantized model is bit-identical to the full pass",
          "[model][quant]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  for (const char* file :
       {"synthetic-2l-q8_0.gguf", "synthetic-2l-q4_0.gguf", "synthetic-2l-q4_1.gguf"}) {
    auto model = load(file);
    const auto full = forward(*model, tokens);
    auto cache = model->new_cache();
    const std::size_t prompt = 8;
    const auto prefix = model->forward_prefill(cache, std::span(tokens).first(prompt));
    REQUIRE(prefix);
    std::vector<float> got(prefix->begin(), prefix->end());
    for (std::size_t i = prompt; i < tokens.size(); ++i) {
      const auto step = model->forward_one(cache, tokens[i]);
      REQUIRE(step);
      got.insert(got.end(), step->begin(), step->end());
    }
    INFO(file);
    CHECK(got == full);
  }
}

TEST_CASE("quantized decode allocates nothing and the arena is sized exactly", "[model][quant]") {
  auto model = load("synthetic-2l-q8_0.gguf", false, 8, 24);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto cache = model->new_cache();
  REQUIRE(model->forward_prefill(cache, std::span(tokens).first(5)));
  const auto before = axon::test::allocation_count();
  REQUIRE(model->forward_one(cache, tokens[5]));
  CHECK(axon::test::allocation_count() == before);

  // Fill the cache so the last prefill is a full max_seq chunk attending over all of max_ctx,
  // then run the cache-free pass at max_seq. Together they reach the larger plan.
  REQUIRE(model->forward_prefill(cache, std::span(tokens).subspan(6, 2)));
  REQUIRE(model->forward_prefill(cache, std::span(tokens).subspan(8, 8)));
  REQUIRE(model->forward_prefill(cache, std::span(tokens).subspan(16, 8)));
  REQUIRE(cache.pos() == 24);
  REQUIRE(model->forward(std::span(tokens).first(8)));
  CHECK(model->arena_high_water() == model->arena_capacity());
}
