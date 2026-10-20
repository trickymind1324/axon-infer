// The NEON table against the scalar table through the whole model. Quantized files run every
// matmul on an integer path that is bit-identical by construction. The NEON attention reduces
// its scores in a different order, a perturbation the activation quantization only shows when
// a value lands on a rounding boundary, which in this small model none does, so the quantized
// logits are equal here. The F32 file goes through the float matmuls, whose reduction order
// differs, so it is held to a tolerance that was measured first (the WARN line) and then locked.

#if defined(AXON_HAVE_NEON)

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

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file,
                                              axon::kernels::KernelSet set) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/" + file),
                                             {.max_seq = 32, .max_ctx = 32, .kernels = set});
  if (!model) {
    if (model.error().code == axon::ErrorCode::Unsupported) {
      SKIP(model.error().message);
    }
    FAIL(model.error().message);
  }
  return std::move(*model);
}

std::vector<float> forward(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens) {
  const auto logits = model.forward(tokens);
  REQUIRE(logits);
  return {logits->begin(), logits->end()};
}

std::vector<float> stepped(axon::model::LlamaModel& model, std::span<const std::int32_t> tokens) {
  auto cache = model.new_cache();
  const auto first = model.forward_prefill(cache, tokens.first(8));
  REQUIRE(first);
  std::vector<float> out(first->begin(), first->end());
  for (std::size_t i = 8; i < tokens.size(); ++i) {
    const auto step = model.forward_one(cache, tokens[i]);
    REQUIRE(step);
    out.insert(out.end(), step->begin(), step->end());
  }
  return out;
}

}  // namespace

TEST_CASE("quantized models give bit-identical logits on the NEON and scalar tables",
          "[model][neon]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  for (const char* file :
       {"synthetic-2l-q8_0.gguf", "synthetic-2l-q4_0.gguf", "synthetic-2l-q4_1.gguf"}) {
    auto scalar = load(file, axon::kernels::KernelSet::Scalar);
    auto neon = load(file, axon::kernels::KernelSet::Neon);
    REQUIRE(axon::kernels::name(neon->table()) == "neon");
    const auto reference = forward(*scalar, tokens);
    INFO(file);
    CHECK(forward(*neon, tokens) == reference);
    CHECK(stepped(*neon, tokens) == reference);
  }
}

TEST_CASE("the F32 model stays within the locked tolerance on the NEON table", "[model][neon]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto scalar = load("synthetic-2l.gguf", axon::kernels::KernelSet::Scalar);
  auto neon = load("synthetic-2l.gguf", axon::kernels::KernelSet::Neon);
  const auto reference = forward(*scalar, tokens);
  const auto got = forward(*neon, tokens);
  float range = 0.0F;
  for (const float v : reference) {
    range = std::max(range, std::abs(v));
  }
  const float diff = axon::test::max_abs_diff(got, reference);
  WARN("F32 synthetic model, NEON against scalar: max abs logit difference "
       << diff << " over a logit range of " << range << ", ratio " << diff / range);
  // Measured 2026-10-06: ratio 1.90e-7. Locked at four times that.
  CHECK(diff <= 8e-7F * range);
  CHECK(stepped(*neon, tokens) == got);  // cache identity holds on the NEON table too
}

#endif
