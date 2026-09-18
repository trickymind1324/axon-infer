#include "core/gguf.h"

#include "fixtures.h"
#include "model/config.h"
#include "model/weights.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

axon::GGUFFile open_fixture(const std::string& name) {
  auto file = axon::GGUFFile::open(axon::test::fixture_path("gguf/" + name));
  if (!file) {
    FAIL("cannot open fixture " << name << ": " << file.error().message);
  }
  return std::move(*file);
}

axon::Expected<axon::model::LlamaWeights> bind_bytes(const std::vector<std::byte>& bytes) {
  auto file = axon::GGUFFile::parse(bytes);
  if (!file) {
    FAIL("patched file does not parse: " << file.error().message);
  }
  auto cfg = axon::model::LlamaConfig::from_gguf(*file);
  if (!cfg) {
    FAIL("patched file has no config: " << cfg.error().message);
  }
  return axon::model::LlamaWeights::bind(*file, *cfg);
}

// Offset of the tensor-info entry for `name` in the header.
std::size_t find_name(const std::vector<std::byte>& bytes, const std::string& name) {
  const auto* begin = reinterpret_cast<const char*>(bytes.data());
  const std::string_view view(begin, bytes.size());
  const auto pos = view.find(name);
  REQUIRE(pos != std::string_view::npos);
  return pos;
}

}  // namespace

TEST_CASE("binding the synthetic model", "[model][weights]") {
  const auto file = open_fixture("synthetic-2l.gguf");
  const auto cfg = axon::model::LlamaConfig::from_gguf(file);
  REQUIRE(cfg);
  const auto w = axon::model::LlamaWeights::bind(file, *cfg);
  REQUIRE(w);
  REQUIRE(w->layers.size() == 2);
  CHECK(w->token_embd.type == axon::DType::F32);
  CHECK(w->token_embd.ne0 == 64);
  CHECK(w->token_embd.ne1 == 512);
  CHECK(w->output.data == w->token_embd.data);  // tied
  CHECK(w->output_norm.ne0 == 64);
  CHECK_FALSE(w->rope_freqs);
  const auto& layer = w->layers[1];
  CHECK(layer.wq.ne0 == 64);
  CHECK(layer.wq.ne1 == 64);
  CHECK(layer.wk.ne1 == 32);
  CHECK(layer.wv.ne1 == 32);
  CHECK(layer.wo.ne0 == 64);
  CHECK(layer.w_gate.ne1 == 160);
  CHECK(layer.w_down.ne0 == 160);
  CHECK(layer.w_down.ne1 == 64);
  CHECK_FALSE(layer.bq);
  CHECK(layer.attn_norm.data != layer.ffn_norm.data);
}

TEST_CASE("quantized weights are refused until their kernels exist", "[model][weights]") {
  const auto file = open_fixture("synthetic-2l-q8_0.gguf");
  const auto cfg = axon::model::LlamaConfig::from_gguf(file);
  REQUIRE(cfg);
  const auto w = axon::model::LlamaWeights::bind(file, *cfg);
  REQUIRE_FALSE(w);
  CHECK(w.error().code == axon::ErrorCode::Unsupported);
  CHECK_THAT(w.error().message, ContainsSubstring("dtype Q8_0 has no kernels yet"));
}

TEST_CASE("a transposed tensor is reported with both shapes", "[model][weights]") {
  auto bytes = axon::test::read_fixture("gguf/synthetic-2l.gguf");
  const std::string name = "blk.0.ffn_down.weight";
  // Tensor info layout: name, u32 n_dims, u64 dims[n_dims], ...; swap the two dims.
  const std::size_t dims_at = find_name(bytes, name) + name.size() + 4;
  std::uint64_t d0 = 0;
  std::uint64_t d1 = 0;
  std::memcpy(&d0, bytes.data() + dims_at, 8);
  std::memcpy(&d1, bytes.data() + dims_at + 8, 8);
  REQUIRE(d0 == 160);
  REQUIRE(d1 == 64);
  std::memcpy(bytes.data() + dims_at, &d1, 8);
  std::memcpy(bytes.data() + dims_at + 8, &d0, 8);

  const auto w = bind_bytes(bytes);
  REQUIRE_FALSE(w);
  CHECK(w.error().code == axon::ErrorCode::InvalidArgument);
  CHECK(w.error().message == "blk.0.ffn_down.weight: expected shape 160x64, found 64x160");
}

TEST_CASE("a missing tensor is named", "[model][weights]") {
  auto bytes = axon::test::read_fixture("gguf/synthetic-2l.gguf");
  const std::string name = "blk.1.ffn_up.weight";
  const std::size_t at = find_name(bytes, name);
  bytes[at + name.size() - 8] = std::byte{'x'};  // "blk.1.ffn_ux.weight"
  const auto w = bind_bytes(bytes);
  REQUIRE_FALSE(w);
  CHECK(w.error().message == "blk.1.ffn_up.weight: missing");
}
