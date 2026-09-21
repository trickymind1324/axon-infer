#include "core/axt.h"

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

std::unique_ptr<axon::model::LlamaModel> load_synthetic(std::size_t max_seq = 32) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/synthetic-2l.gguf"),
                                             {.max_seq = max_seq, .max_ctx = max_seq});
  if (!model) {
    FAIL("cannot load the synthetic model: " << model.error().message);
  }
  return std::move(*model);
}

std::vector<float> run(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens) {
  const auto logits = model.forward(tokens);
  if (!logits) {
    FAIL("forward failed: " << logits.error().message);
  }
  return {logits->begin(), logits->end()};
}

}  // namespace

TEST_CASE("synthetic forward matches the NumPy reference", "[model][forward]") {
  auto model = load_synthetic();
  CHECK(model->config().n_vocab == 512);
  CHECK(model->tokenizer().vocab_size() == 512);
  for (const char* name : {"synthetic-2l", "synthetic-2l.short"}) {
    const std::string base = name;
    const auto tokens = axon::test::load_axt(base + ".tokens", "model").as_i32();
    const auto expected = axon::test::load_axt(base + ".logits", "model");
    REQUIRE(expected.dim(0) == tokens.size());
    REQUIRE(expected.dim(1) == 512);
    const auto logits = run(*model, tokens);
    INFO(name);
    CHECK(axon::test::max_abs_diff(logits, expected.as_f32()) < 1e-4F);
  }
}

TEST_CASE("a prefix sees the same logits as the full sequence", "[model][forward]") {
  auto model = load_synthetic();
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto full = run(*model, tokens);
  const std::size_t n = 10;
  const auto prefix = run(*model, std::span(tokens).first(n));
  const std::vector<float> head(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n * 512));
  CHECK(prefix == head);
}

TEST_CASE("forward is deterministic and reuses the arena", "[model][forward]") {
  auto model = load_synthetic();
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto first = run(*model, tokens);
  const auto second = run(*model, tokens);
  CHECK(first == second);
  CHECK(model->arena_capacity() == model->config().activation_bytes(32, 32));
}

TEST_CASE("the arena is sized exactly for max_seq", "[model][forward]") {
  const std::size_t max_seq = 19;
  auto model = load_synthetic(max_seq);
  std::vector<std::int32_t> tokens(max_seq);
  for (std::size_t i = 0; i < max_seq; ++i) {
    tokens[i] = static_cast<std::int32_t>((i * 37) % 512);
  }
  run(*model, tokens);
  CHECK(model->arena_high_water() == model->config().activation_bytes(max_seq, max_seq));
  CHECK(model->arena_high_water() == model->arena_capacity());
}

TEST_CASE("forward rejects bad input", "[model][forward]") {
  auto model = load_synthetic(4);
  const std::vector<std::int32_t> long_input{1, 2, 3, 4, 5};
  const auto too_long = model->forward(long_input);
  REQUIRE_FALSE(too_long);
  CHECK(too_long.error().message == "forward: 5 tokens exceed max_seq 4");

  const std::vector<std::int32_t> out_of_range{1, 512};
  const auto bad_id = model->forward(out_of_range);
  REQUIRE_FALSE(bad_id);
  CHECK(bad_id.error().message == "forward: token id 512 outside [0, 512)");

  const auto empty = model->forward(std::span<const std::int32_t>{});
  REQUIRE_FALSE(empty);
  CHECK_THAT(empty.error().message, ContainsSubstring("empty"));

  const auto zero = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/synthetic-2l.gguf"), {.max_seq = 0, .max_ctx = 8});
  REQUIRE_FALSE(zero);
  CHECK_THAT(zero.error().message, ContainsSubstring("max_seq"));
}

TEST_CASE("a quantized file loads", "[model][forward]") {
  const auto model = axon::model::LlamaModel::load(
      axon::test::fixture_path("gguf/synthetic-2l-q4_0.gguf"), {.max_seq = 8, .max_ctx = 8});
  REQUIRE(model);
  CHECK((*model)->weights().token_embd.type == axon::DType::Q4_0);
}

TEST_CASE("forward performs no heap allocation", "[model][forward]") {
  auto model = load_synthetic();
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  run(*model, tokens);  // first call: nothing lazy should remain, but be safe
  const auto before = axon::test::allocation_count();
  const auto logits = model->forward(tokens);
  const auto after = axon::test::allocation_count();
  REQUIRE(logits);
  CHECK(after == before);
}

TEST_CASE("activation dumps are written as axt files", "[model][forward]") {
  auto model = load_synthetic();
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto dir = std::filesystem::temp_directory_path() / "axon_dump_test";
  std::filesystem::create_directories(dir);
  const auto logits = model->forward(tokens, {.dump_layer = 1, .dump_dir = dir});
  REQUIRE(logits);
  const std::vector<float> returned(logits->begin(), logits->end());

  for (const char* name : {"embed", "x_in", "attn_norm_out", "q", "k", "v", "attn_out", "x_mid",
                           "ffn_norm_out", "ffn_out", "x_out", "logits"}) {
    INFO(name);
    CHECK(std::filesystem::exists(dir / (std::string(name) + ".axt")));
  }
  const auto k = axon::read_axt(dir / "k.axt");
  REQUIRE(k);
  CHECK(k->dims == std::vector<std::uint64_t>{24, 32});
  const auto dumped = axon::read_axt(dir / "logits.axt");
  REQUIRE(dumped);
  CHECK(dumped->dims == std::vector<std::uint64_t>{24, 512});
  CHECK(dumped->as_f32() == returned);

  const auto unwritable = model->forward(tokens, {.dump_layer = 0, .dump_dir = "/nonexistent"});
  REQUIRE_FALSE(unwritable);
  CHECK(unwritable.error().code == axon::ErrorCode::Io);
  std::filesystem::remove_all(dir);
}
