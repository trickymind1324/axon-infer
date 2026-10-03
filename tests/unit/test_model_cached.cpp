// The cached forward must reproduce the cache-free reference bit for bit, however the
// tokens are split between prefill chunks and single-token steps.

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "model/kv_cache.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

constexpr std::size_t kVocab = 512;

std::unique_ptr<axon::model::LlamaModel> load_synthetic(std::size_t max_seq, std::size_t max_ctx) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = max_seq, .max_ctx = max_ctx});
  if (!model) {
    FAIL("cannot load the synthetic model: " << model.error().message);
  }
  return std::move(*model);
}

std::vector<float> copy(const axon::Expected<std::span<const float>>& logits) {
  if (!logits) {
    FAIL("forward failed: " << logits.error().message);
  }
  return {logits->begin(), logits->end()};
}

std::vector<float> reference(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens) {
  return copy(model.forward(tokens));
}

std::vector<float> last_row(const std::vector<float>& logits) {
  return {logits.end() - static_cast<std::ptrdiff_t>(kVocab), logits.end()};
}

}  // namespace

TEST_CASE("prefill over an empty cache equals the cache-free forward", "[model][cache]") {
  auto model = load_synthetic(32, 32);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto expected = reference(*model, tokens);
  auto cache = model->new_cache();
  const auto got = copy(model->forward_prefill(cache, tokens));
  CHECK(got == expected);
  CHECK(cache.pos() == tokens.size());
}

TEST_CASE("decoding one token at a time is bit-identical to the full pass", "[model][cache]") {
  auto model = load_synthetic(32, 32);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto cache = model->new_cache();
  const std::size_t prompt = 8;
  const auto prefix = copy(model->forward_prefill(cache, std::span(tokens).first(prompt)));
  const auto full = reference(*model, tokens);
  const std::vector<float> full_head(full.begin(),
                                     full.begin() + static_cast<std::ptrdiff_t>(prompt * kVocab));
  CHECK(prefix == full_head);
  for (std::size_t i = prompt; i < tokens.size(); ++i) {
    const auto step = copy(model->forward_one(cache, tokens[i]));
    REQUIRE(step.size() == kVocab);
    const auto expected = last_row(reference(*model, std::span(tokens).first(i + 1)));
    INFO("step " << i);
    CHECK(step == expected);
    CHECK(cache.pos() == i + 1);
  }
}

TEST_CASE("chunked prefill equals one prefill", "[model][cache]") {
  auto model = load_synthetic(32, 32);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto expected = reference(*model, tokens);
  auto cache = model->new_cache();
  std::vector<float> got;
  for (const std::size_t chunk : {std::size_t{5}, std::size_t{7}, std::size_t{12}}) {
    const auto part =
        copy(model->forward_prefill(cache, std::span(tokens).subspan(cache.pos(), chunk)));
    got.insert(got.end(), part.begin(), part.end());
  }
  CHECK(got == expected);
}

TEST_CASE("truncating the cache and refeeding reproduces the straight run", "[model][cache]") {
  auto model = load_synthetic(32, 32);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto cache = model->new_cache();
  copy(model->forward_prefill(cache, tokens));
  cache.truncate(12);
  const auto tail = copy(model->forward_prefill(cache, std::span(tokens).subspan(12)));
  const auto expected = reference(*model, tokens);
  const std::vector<float> expected_tail(
      expected.begin() + static_cast<std::ptrdiff_t>(12 * kVocab), expected.end());
  CHECK(tail == expected_tail);
  CHECK(cache.pos() == tokens.size());
}

TEST_CASE("cached forward rejects bad input", "[model][cache]") {
  auto model = load_synthetic(4, 6);
  auto cache = model->new_cache();
  CHECK(cache.max_ctx() == 6);
  const std::vector<std::int32_t> four{1, 2, 3, 4};
  REQUIRE(model->forward_prefill(cache, four));
  const std::vector<std::int32_t> three{5, 6, 7};
  const auto overflow = model->forward_prefill(cache, three);
  REQUIRE_FALSE(overflow);
  CHECK(overflow.error().message == "context overflow: 4 cached + 3 new tokens exceed 6");
  CHECK(cache.pos() == 4);

  const std::vector<std::int32_t> five{1, 2, 3, 4, 5};
  auto fresh = model->new_cache();
  const auto too_many = model->forward_prefill(fresh, five);
  REQUIRE_FALSE(too_many);
  CHECK(too_many.error().message == "forward: 5 tokens exceed max_seq 4");

  const auto empty = model->forward_prefill(fresh, std::span<const std::int32_t>{});
  REQUIRE_FALSE(empty);
  CHECK_THAT(empty.error().message, ContainsSubstring("empty"));

  const auto bad_id = model->forward_one(fresh, 512);
  REQUIRE_FALSE(bad_id);
  CHECK(bad_id.error().message == "forward: token id 512 outside [0, 512)");

  auto foreign_cfg = model->config();
  foreign_cfg.n_layer = 3;
  axon::model::KVCache foreign(foreign_cfg, 6);
  const auto mismatch = model->forward_one(foreign, 1);
  REQUIRE_FALSE(mismatch);
  CHECK(mismatch.error().message == "forward: the cache was built for a different model");
}

TEST_CASE("load clamps max_ctx to the training context and refuses a tiny one", "[model][cache]") {
  auto clamped = load_synthetic(8, 100000);
  CHECK(clamped->max_ctx() == 128);
  CHECK(clamped->max_seq() == 8);
  auto both = load_synthetic(256, 2048);
  CHECK(both->max_ctx() == 128);
  CHECK(both->max_seq() == 128);
  const auto tiny = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/synthetic-2l.gguf"), {.max_seq = 16, .max_ctx = 8});
  REQUIRE_FALSE(tiny);
  CHECK(tiny.error().message == "max_ctx 8 is smaller than max_seq 16");
}

TEST_CASE("forward_one performs no heap allocation", "[model][cache]") {
  auto model = load_synthetic(8, 32);
  auto cache = model->new_cache();
  const std::vector<std::int32_t> prompt{3, 1, 4, 1, 5};
  REQUIRE(model->forward_prefill(cache, prompt));
  const auto before = axon::test::allocation_count();
  const auto logits = model->forward_one(cache, 9);
  const auto after = axon::test::allocation_count();
  REQUIRE(logits);
  CHECK(after == before);
}

TEST_CASE("the arena covers both forward paths exactly", "[model][cache]") {
  const std::size_t max_seq = 8;
  const std::size_t max_ctx = 24;
  auto model = load_synthetic(max_seq, max_ctx);
  CHECK(model->arena_capacity() ==
        model->config().activation_bytes(max_seq, max_ctx,
                                         static_cast<std::size_t>(model->n_threads())));
  std::vector<std::int32_t> tokens(max_ctx);
  for (std::size_t i = 0; i < max_ctx; ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 53) % kVocab);
  }
  // Fill the whole cache in max_seq chunks, so the last chunk attends over max_ctx positions,
  // and run the cache-free pass at max_seq. Together they touch the larger of the two plans.
  auto cache = model->new_cache();
  for (std::size_t offset = 0; offset < max_ctx; offset += max_seq) {
    REQUIRE(model->forward_prefill(cache, std::span(tokens).subspan(offset, max_seq)));
  }
  REQUIRE(model->forward(std::span(tokens).first(max_seq)));
  CHECK(model->arena_high_water() == model->arena_capacity());
}
