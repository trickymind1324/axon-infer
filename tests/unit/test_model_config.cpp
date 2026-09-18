#include "core/gguf.h"

#include "fixtures.h"
#include "gguf_builder.h"
#include "model/config.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <string>

using Catch::Matchers::ContainsSubstring;

namespace {

axon::GGUFFile open_fixture(const std::string& name) {
  auto file = axon::GGUFFile::open(axon::test::fixture_path("gguf/" + name));
  if (!file) {
    FAIL("cannot open fixture " << name << ": " << file.error().message);
  }
  return std::move(*file);
}

axon::test::GGUFBuilder llama_metadata() {
  return axon::test::GGUFBuilder()
      .string("general.architecture", "llama")
      .u32("llama.block_count", 2)
      .u32("llama.embedding_length", 64)
      .u32("llama.attention.head_count", 4)
      .u32("llama.attention.head_count_kv", 2)
      .u32("llama.feed_forward_length", 160)
      .u32("llama.context_length", 128);
}

axon::Expected<axon::model::LlamaConfig> config_of(const std::vector<std::byte>& bytes) {
  auto file = axon::GGUFFile::parse(bytes);
  if (!file) {
    FAIL("builder produced an unparseable file: " << file.error().message);
  }
  return axon::model::LlamaConfig::from_gguf(*file);
}

}  // namespace

TEST_CASE("config from the synthetic model", "[model][config]") {
  const auto file = open_fixture("synthetic-2l.gguf");
  const auto cfg = axon::model::LlamaConfig::from_gguf(file);
  REQUIRE(cfg);
  CHECK(cfg->arch == "llama");
  CHECK(cfg->n_layer == 2);
  CHECK(cfg->n_embd == 64);
  CHECK(cfg->n_head == 4);
  CHECK(cfg->n_head_kv == 2);
  CHECK(cfg->n_ff == 160);
  CHECK(cfg->n_vocab == 512);
  CHECK(cfg->n_ctx_train == 128);
  CHECK(cfg->head_dim == 16);
  CHECK(cfg->rotary_dim == 16);
  CHECK(cfg->rope_theta == 10000.0F);
  CHECK(std::abs(cfg->rms_eps - 1e-5F) < 1e-9F);
  CHECK(cfg->rope_style == axon::kernels::RopeStyle::Normal);
  CHECK(cfg->rope_scaling == axon::model::RopeScaling::None);
  CHECK(cfg->tie_embeddings);
  CHECK_FALSE(cfg->attn_bias);
  CHECK(cfg->n_embd_q() == 64);
  CHECK(cfg->n_embd_kv() == 32);
  CHECK(cfg->describe() ==
        "llama layers 2 embd 64 heads 4 kv 2 ff 160 vocab 512 ctx 128 rope normal 10000 tied");
}

TEST_CASE("unsupported architecture is rejected by name", "[model][config]") {
  const auto file = open_fixture("tiny.gguf");
  const auto cfg = axon::model::LlamaConfig::from_gguf(file);
  REQUIRE_FALSE(cfg);
  CHECK(cfg.error().code == axon::ErrorCode::Unsupported);
  CHECK(cfg.error().message == "general.architecture: expected llama or qwen2, found 'axon-test'");
}

TEST_CASE("missing keys are named", "[model][config]") {
  auto no_layers = axon::test::GGUFBuilder()
                       .string("general.architecture", "llama")
                       .u32("llama.embedding_length", 64);
  const auto cfg = config_of(no_layers.build());
  REQUIRE_FALSE(cfg);
  CHECK_THAT(cfg.error().message, ContainsSubstring("llama.block_count"));

  const auto no_tensor = config_of(llama_metadata().build());
  REQUIRE_FALSE(no_tensor);
  CHECK(no_tensor.error().message == "token_embd.weight: missing");
}

TEST_CASE("inconsistent head and rope counts are rejected", "[model][config]") {
  const auto bad_kv = config_of(axon::test::GGUFBuilder()
                                    .string("general.architecture", "llama")
                                    .u32("llama.block_count", 2)
                                    .u32("llama.embedding_length", 64)
                                    .u32("llama.attention.head_count", 4)
                                    .u32("llama.attention.head_count_kv", 3)
                                    .u32("llama.feed_forward_length", 160)
                                    .u32("llama.context_length", 128)
                                    .build());
  REQUIRE_FALSE(bad_kv);
  CHECK(bad_kv.error().message ==
        "llama.attention.head_count_kv: expected head_count (4) to be a multiple of "
        "head_count_kv, found 3");

  const auto bad_rope = config_of(llama_metadata().u32("llama.rope.dimension_count", 24).build());
  REQUIRE_FALSE(bad_rope);
  CHECK(bad_rope.error().message ==
        "llama.rope.dimension_count: expected an even value up to the head size 16, found 24");
}

TEST_CASE("qwen2 selects NeoX rope", "[model][config]") {
  const auto cfg = config_of(axon::test::GGUFBuilder()
                                 .string("general.architecture", "qwen2")
                                 .u32("qwen2.block_count", 1)
                                 .u32("qwen2.embedding_length", 32)
                                 .u32("qwen2.attention.head_count", 2)
                                 .u32("qwen2.feed_forward_length", 64)
                                 .u32("qwen2.context_length", 64)
                                 .f32("qwen2.rope.freq_base", 1000000.0F)
                                 .build());
  // No tensors, so it fails at the embedding; the architecture branch ran before that.
  REQUIRE_FALSE(cfg);
  CHECK(cfg.error().message == "token_embd.weight: missing");
}

TEST_CASE("activation bytes grow with the sequence", "[model][config]") {
  const auto file = open_fixture("synthetic-2l.gguf");
  const auto cfg = axon::model::LlamaConfig::from_gguf(file);
  REQUIRE(cfg);
  const auto one = cfg->activation_bytes(1);
  const auto many = cfg->activation_bytes(64);
  CHECK(one > 0);
  CHECK(many > one);
  CHECK(one % 64 == 0);
  // Ten buffers for one token: 4 x 64 floats, 2 x 32, 2 x 160, 1 x 1, 1 x 512.
  CHECK(one == 4 * 256 + 2 * 128 + 2 * 640 + 64 + 2048);
}
