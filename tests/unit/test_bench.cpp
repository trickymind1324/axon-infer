#include "engine/bench.h"
#include "fixtures.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <vector>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("the benchmark reports a consistent result and a complete JSON object",
          "[engine][bench]") {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = 16, .max_ctx = 64, .n_threads = 2});
  REQUIRE(model);
  std::vector<std::int32_t> prompt(16);
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    prompt[i] = static_cast<std::int32_t>((i * 31 + 7) % 512);
  }
  int progress_calls = 0;
  const auto result =
      axon::engine::run_benchmark(**model, prompt, {.generate = 4, .runs = 3},
                                  [&](const axon::engine::BenchRun& run, int index) {
                                    CHECK(run.prefill_s > 0.0);
                                    CHECK(run.decode_s > 0.0);
                                    CHECK(index == progress_calls);
                                    ++progress_calls;
                                  });
  REQUIRE(result);
  CHECK(progress_calls == 3);
  CHECK(result->runs.size() == 3);
  CHECK(result->prompt_tokens == 16);
  CHECK(result->generated_tokens == 4);
  CHECK(result->threads == 2);
  CHECK(result->context == 64);
  CHECK(result->prefill_tok_s_median <= result->prefill_tok_s_best);
  CHECK(result->decode_tok_s_median <= result->decode_tok_s_best);
  CHECK(result->weight_bytes_per_token == (*model)->weight_bytes_per_token());
  // Two layers, 32 kv floats, 4 bytes, mean position 16 + 1.5.
  CHECK(result->cache_bytes_per_token == 2.0 * 2 * 32 * 4 * 17.5);
  CHECK(result->weight_gb_s_median > 0.0);

  const auto json = axon::engine::bench_json(*result, "synthetic.gguf", "two layers");
  for (const char* key :
       {"\"model_file\"", "\"threads\": 2", "\"prompt_tokens\": 16", "\"generated_tokens\": 4",
        "\"runs\": 3", "\"prefill_tok_s\"", "\"decode_tok_s\"", "\"weight_bytes_per_token\"",
        "\"cache_bytes_per_token\"", "\"weight_gb_s\"", "\"peak_rss_mb\"", "\"load_average\"",
        "\"measurements\": ["}) {
    INFO(key);
    CHECK_THAT(json, ContainsSubstring(key));
  }

  const auto too_long = axon::engine::run_benchmark(**model, prompt, {.generate = 60, .runs = 1});
  REQUIRE_FALSE(too_long);
  CHECK_THAT(too_long.error().message, ContainsSubstring("exceed the context of 64"));
}
