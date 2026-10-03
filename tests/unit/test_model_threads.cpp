// Threading never changes a result: work is split so that no partial sum crosses a thread, so
// the logits at any thread count equal the single-threaded logits bit for bit.

#include "core/cpu_info.h"

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file, int threads,
                                              std::size_t max_seq = 32, std::size_t max_ctx = 32) {
  auto model =
      axon::model::LlamaModel::load(axon::test::fixture_path("gguf/" + file),
                                    {.max_seq = max_seq, .max_ctx = max_ctx, .n_threads = threads});
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

// Prefill a prompt, then decode the rest one token at a time, concatenating every row.
std::vector<float> cached(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens,
                          std::size_t prompt) {
  auto cache = model.new_cache();
  const auto first = model.forward_prefill(cache, tokens.first(prompt));
  REQUIRE(first);
  std::vector<float> out(first->begin(), first->end());
  for (std::size_t i = prompt; i < tokens.size(); ++i) {
    const auto step = model.forward_one(cache, tokens[i]);
    REQUIRE(step);
    out.insert(out.end(), step->begin(), step->end());
  }
  return out;
}

}  // namespace

TEST_CASE("logits are bit-identical at every thread count", "[model][threads]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf", "synthetic-2l-q4_0.gguf",
                           "synthetic-2l-q4_1.gguf"}) {
    auto single = load(file, 1);
    CHECK(single->n_threads() == 1);
    const auto full = forward(*single, tokens);
    const auto stepped = cached(*single, tokens, 8);
    CHECK(stepped == full);
    // Odd counts give uneven splits of the 64 output columns and the 4 heads.
    for (const int threads : {2, 3, 7}) {
      auto model = load(file, threads);
      CHECK(model->n_threads() == threads);
      INFO(file << " at " << threads << " threads");
      CHECK(forward(*model, tokens) == full);
      CHECK(cached(*model, tokens, 8) == full);
    }
  }
}

TEST_CASE("more threads than heads or rows still works and still matches", "[model][threads]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto single = load("synthetic-2l-q8_0.gguf", 1);
  const auto full = forward(*single, tokens);
  auto many = load("synthetic-2l-q8_0.gguf", 13);  // 4 heads, 32 kv columns
  CHECK(forward(*many, tokens) == full);
  CHECK(cached(*many, std::span(tokens).first(3), 1) ==
        std::vector<float>(full.begin(), full.begin() + 3 * 512));
}

TEST_CASE("zero threads means one per physical core", "[model][threads]") {
  auto model = load("synthetic-2l.gguf", 0);
  CHECK(model->n_threads() == axon::cpu_info().physical_cores);
  CHECK(model->arena_capacity() ==
        model->config().activation_bytes(32, 32, static_cast<std::size_t>(model->n_threads())));
}

TEST_CASE("threaded decode allocates nothing", "[model][threads]") {
  auto model = load("synthetic-2l-q4_0.gguf", 4, 8, 32);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto cache = model->new_cache();
  REQUIRE(model->forward_prefill(cache, std::span(tokens).first(6)));
  REQUIRE(model->forward_one(cache, tokens[6]));
  const auto before = axon::test::allocation_count();
  for (std::size_t i = 7; i < 12; ++i) {
    REQUIRE(model->forward_one(cache, tokens[i]));
  }
  CHECK(axon::test::allocation_count() == before);
  CHECK(model->weight_bytes_per_token() > 0);
}
