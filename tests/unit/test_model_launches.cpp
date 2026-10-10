// Launches per token: the count of parallel_for calls a decode step issues. Five per layer
// (q/k/v, attention, output projection, gate/up, down) plus the output matmul, with activation
// quantization inline. Prefill adds one quantization launch per quantized input.

#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <vector>

TEST_CASE("decode issues five launches per layer plus one", "[model][launches]") {
  for (const char* file : {"synthetic-2l.gguf", "synthetic-2l-q8_0.gguf"}) {
    auto model =
        axon::model::LlamaModel::load(axon::test::fixture_path(std::string("gguf/") + file),
                                      {.max_seq = 8, .max_ctx = 32, .n_threads = 2});
    REQUIRE(model);
    const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
    auto cache = (*model)->new_cache();
    REQUIRE((*model)->forward_prefill(cache, std::span(tokens).first(4)));
    const auto before = (*model)->pool().launches();
    REQUIRE((*model)->forward_one(cache, tokens[4]));
    const auto per_token = (*model)->pool().launches() - before;
    INFO(file);
    CHECK(per_token == 2 * 5 + 1);
  }
}

TEST_CASE("prefill adds one quantization launch per quantized input", "[model][launches]") {
  auto model =
      axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l-q4_0.gguf"),
                                    {.max_seq = 8, .max_ctx = 32, .n_threads = 2});
  REQUIRE(model);
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto cache = (*model)->new_cache();
  const auto before = (*model)->pool().launches();
  REQUIRE((*model)->forward_prefill(cache, std::span(tokens).first(4)));
  // Per layer the four quantized inputs each cost one launch: the normed row for q/k/v, the
  // attention output for the output projection, the normed row for gate/up, and the gate
  // product for down. Plus the output matmul and its own quantization.
  const auto launches = (*model)->pool().launches() - before;
  INFO("prefill launches " << launches);
  CHECK(launches == 2 * (5 + 4) + 1 + 1);
}
