// The fp16 key/value cache. Against the F32 cache its logits differ by the rounding of each
// key and value, bounded by a tolerance that was measured first and then locked. On its
// own it keeps the identities the F32 cache has: the same logits however the tokens are split
// into prefill chunks and single steps, at any thread count, and no heap allocation in decode.

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "kernels/kernels.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel>
load(const std::string& file, axon::model::CacheType cache, int threads = 1,
     axon::kernels::KernelSet set = axon::kernels::KernelSet::Auto) {
  auto model = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/" + file),
      {.max_seq = 8, .max_ctx = 32, .n_threads = threads, .kernels = set, .cache = cache});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

std::vector<float> stepped(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens,
                           std::size_t prompt) {
  auto cache = model.new_cache();
  std::vector<float> out;
  for (std::size_t offset = 0; offset < prompt; offset += model.max_seq()) {
    const std::size_t n = std::min(model.max_seq(), prompt - offset);
    const auto part = model.forward_prefill(cache, tokens.subspan(offset, n));
    REQUIRE(part);
    out.insert(out.end(), part->begin(), part->end());
  }
  for (std::size_t i = prompt; i < tokens.size(); ++i) {
    const auto step = model.forward_one(cache, tokens[i]);
    REQUIRE(step);
    out.insert(out.end(), step->begin(), step->end());
  }
  return out;
}

float range_of(const std::vector<float>& v) {
  float r = 0.0F;
  for (const float x : v) {
    r = std::max(r, std::abs(x));
  }
  return r;
}

}  // namespace

TEST_CASE("the fp16 cache stays within the locked tolerance of the F32 cache",
          "[model][cache_f16]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  float worst_ratio = 0.0F;
  for (const char* file :
       {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf", "synthetic-2l-q4_0.gguf"}) {
    auto full = load(file, axon::model::CacheType::F32);
    auto half = load(file, axon::model::CacheType::F16);
    REQUIRE(half->cache_type() == axon::model::CacheType::F16);
    const auto reference = stepped(*full, tokens, 8);
    const auto got = stepped(*half, tokens, 8);
    const float ratio = axon::test::max_abs_diff(got, reference) / range_of(reference);
    INFO(file << " ratio " << ratio);
    worst_ratio = std::max(worst_ratio, ratio);
    auto cache = half->new_cache();
    CHECK(cache.bytes() == std::size_t{2} * 2 * 32 * 32 * 2);  // half the F32 cache
  }
  // Measured 2026-10-10 on the synthetic models: 2.3e-5 for the F32 weights and 1.7e-3 for the
  // Q8_0 and Q4_0 weights, where the rounded keys and values shift the Q8_0 activation
  // quantization in front of the output projection. Locked at four times the measurement.
  INFO("fp16 cache against F32 cache, max abs logit difference over the logit range: "
       << worst_ratio);
  CHECK(worst_ratio < 7e-3F);
}

TEST_CASE("the fp16 cache keeps its own identities", "[model][cache_f16]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q4_0.gguf"}) {
    auto one = load(file, axon::model::CacheType::F16, 1);
    const auto a = stepped(*one, tokens, 8);
    CHECK(stepped(*one, tokens, 5) == a);   // a different prefill chunking
    CHECK(stepped(*one, tokens, 16) == a);  // two prefill chunks of 8
    auto three = load(file, axon::model::CacheType::F16, 3);
    CHECK(stepped(*three, tokens, 8) == a);
    auto cache = three->new_cache();
    REQUIRE(three->forward_prefill(cache, std::span(tokens).first(6)));
    REQUIRE(three->forward_one(cache, tokens[6]));
    const auto before = axon::test::allocation_count();
    REQUIRE(three->forward_one(cache, tokens[7]));
    CHECK(axon::test::allocation_count() == before);
  }
}

TEST_CASE("the fp16 cache sizes the arena exactly and refuses the other cache type",
          "[model][cache_f16]") {
  std::vector<std::int32_t> tokens(32);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 53) % 512);
  }
  auto half = load("synthetic-2l.gguf", axon::model::CacheType::F16);
  CHECK(half->arena_capacity() ==
        half->config().activation_bytes(8, 32, static_cast<std::size_t>(half->n_threads()),
                                        axon::model::CacheType::F16));
  // Fill the cache in max_seq chunks so the last one attends over the whole context, then run
  // the cache-free pass at max_seq. Together they reach the larger of the two plans.
  auto cache = half->new_cache();
  for (std::size_t offset = 0; offset < 32; offset += 8) {
    REQUIRE(half->forward_prefill(cache, std::span(tokens).subspan(offset, 8)));
  }
  REQUIRE(half->forward(std::span(tokens).first(8)));
  CHECK(half->arena_high_water() == half->arena_capacity());

  auto full = load("synthetic-2l.gguf", axon::model::CacheType::F32);
  auto wrong = axon::model::KVCache(full->config(), 32, axon::model::CacheType::F16);
  const auto refused = full->forward_prefill(wrong, std::span(tokens).first(4));
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == axon::ErrorCode::InvalidArgument);
  auto other = axon::model::KVCache(half->config(), 32, axon::model::CacheType::F32);
  REQUIRE_FALSE(half->forward_prefill(other, std::span(tokens).first(4)));
}

#if defined(AXON_HAVE_NEON)
TEST_CASE("NEON fp16 attention stays within the locked tolerance of scalar",
          "[model][cache_f16][neon]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  float worst = 0.0F;
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf"}) {
    auto scalar = load(file, axon::model::CacheType::F16, 1, axon::kernels::KernelSet::Scalar);
    auto neon = load(file, axon::model::CacheType::F16, 1, axon::kernels::KernelSet::Neon);
    const auto a = stepped(*scalar, tokens, 8);
    const auto b = stepped(*neon, tokens, 8);
    worst = std::max(worst, axon::test::max_abs_diff(a, b) / range_of(a));
  }
  // Measured 2026-10-10: 1.9e-7, the same order as the NEON float matmuls. Locked at four
  // times the measurement.
  INFO("NEON fp16 attention against scalar, max abs logit difference over the logit range: "
       << worst);
  CHECK(worst < 8e-7F);
}
#endif
