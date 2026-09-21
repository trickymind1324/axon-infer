#include "engine/perplexity.h"
#include "fixtures.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

constexpr std::size_t kVocab = 512;

std::unique_ptr<axon::model::LlamaModel> load_synthetic(std::size_t max_seq, std::size_t max_ctx) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = max_seq, .max_ctx = max_ctx});
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

std::vector<std::int32_t> stream(std::size_t n) {
  std::vector<std::int32_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<std::int32_t>((i * 7919 + 13) % kVocab);
  }
  return out;
}

// Independent scoring through the cache-free forward, one chunk at a time.
double reference_nll(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens,
                     std::size_t chunk, std::size_t n_chunks, std::size_t* scored) {
  double nll = 0.0;
  *scored = 0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    const auto window = tokens.subspan(c * chunk, chunk);
    const auto logits = model.forward(window);
    REQUIRE(logits);
    for (std::size_t p = chunk / 2 - 1; p + 1 < chunk; ++p) {
      const float* row = logits->data() + p * kVocab;
      auto max = static_cast<double>(row[0]);
      for (std::size_t j = 0; j < kVocab; ++j) {
        max = std::max(max, static_cast<double>(row[j]));
      }
      double sum = 0.0;
      for (std::size_t j = 0; j < kVocab; ++j) {
        sum += std::exp(static_cast<double>(row[j]) - max);
      }
      nll +=
          max + std::log(sum) - static_cast<double>(row[static_cast<std::size_t>(window[p + 1])]);
      ++*scored;
    }
  }
  return nll;
}

}  // namespace

TEST_CASE("perplexity scores the second half of every chunk", "[engine][perplexity]") {
  auto model = load_synthetic(32, 32);
  const auto tokens = stream(std::size_t{3} * 32 + 5);  // three full chunks and a partial one
  std::vector<std::size_t> seen;
  const auto result = axon::engine::perplexity(*model, tokens, {.chunk_size = 32},
                                               [&](std::size_t c, double) { seen.push_back(c); });
  REQUIRE(result);
  CHECK(result->chunks == 3);
  CHECK(result->scored_tokens == std::size_t{3} * 16);
  CHECK(seen == std::vector<std::size_t>{0, 1, 2});
  CHECK(result->running.size() == 3);
  CHECK(result->perplexity() > 1.0);
  CHECK(std::abs(result->running.back() - result->perplexity()) < 1e-12);

  std::size_t scored = 0;
  const double expected = reference_nll(*model, tokens, 32, 3, &scored);
  CHECK(scored == result->scored_tokens);
  CHECK(std::abs(result->nll_sum - expected) < 1e-9 * std::abs(expected));
}

TEST_CASE("perplexity is the same when the chunk is prefilled in pieces", "[engine][perplexity]") {
  const auto tokens = stream(std::size_t{2} * 32);
  auto whole = load_synthetic(32, 32);
  auto pieces = load_synthetic(8, 32);
  const auto a = axon::engine::perplexity(*whole, tokens, {.chunk_size = 32});
  const auto b = axon::engine::perplexity(*pieces, tokens, {.chunk_size = 32});
  REQUIRE(a);
  REQUIRE(b);
  CHECK(a->nll_sum == b->nll_sum);
  const auto limited =
      axon::engine::perplexity(*pieces, tokens, {.chunk_size = 32, .max_chunks = 1});
  REQUIRE(limited);
  CHECK(limited->chunks == 1);
  CHECK(limited->scored_tokens == 16);
}

TEST_CASE("perplexity rejects impossible settings", "[engine][perplexity]") {
  auto model = load_synthetic(8, 16);
  const auto tokens = stream(64);
  const auto odd = axon::engine::perplexity(*model, tokens, {.chunk_size = 7});
  REQUIRE_FALSE(odd);
  CHECK_THAT(odd.error().message, ContainsSubstring("even"));
  const auto too_long = axon::engine::perplexity(*model, tokens, {.chunk_size = 32});
  REQUIRE_FALSE(too_long);
  CHECK_THAT(too_long.error().message, ContainsSubstring("max_ctx 16"));
  const auto short_stream =
      axon::engine::perplexity(*model, std::span(tokens).first(10), {.chunk_size = 16});
  REQUIRE_FALSE(short_stream);
  CHECK_THAT(short_stream.error().message, ContainsSubstring("no complete chunk"));
}
