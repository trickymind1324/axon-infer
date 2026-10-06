#include "core/cpu_info.h"

#include "fixture_tensor.h"
#include "kernels/kernels.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("kernel sets resolve to named tables", "[kernels][dispatch]") {
  const auto scalar = axon::kernels::select(axon::kernels::KernelSet::Scalar);
  REQUIRE(scalar);
  CHECK(*scalar == &axon::kernels::scalar_kernels());
  CHECK(axon::kernels::name(**scalar) == "scalar");
  CHECK(&axon::kernels::kernels() == &axon::kernels::scalar_kernels());

  const auto best = axon::kernels::select(axon::kernels::KernelSet::Auto);
  REQUIRE(best);
  const auto& cpu = axon::cpu_info();
#if defined(AXON_HAVE_NEON)
  if (cpu.neon && cpu.neon_dotprod) {
    CHECK(axon::kernels::name(**best) == "neon");
    CHECK(*best != *scalar);
    const auto neon = axon::kernels::select(axon::kernels::KernelSet::Neon);
    REQUIRE(neon);
    CHECK(*neon == *best);
  } else {
    CHECK(*best == *scalar);
    const auto neon = axon::kernels::select(axon::kernels::KernelSet::Neon);
    REQUIRE_FALSE(neon);
    CHECK_THAT(neon.error().message, ContainsSubstring("FEAT_DotProd"));
  }
#else
  CHECK(*best == *scalar);
  CHECK_FALSE(axon::kernels::select(axon::kernels::KernelSet::Neon));
#endif
  axon::kernels::KernelSet parsed{};
  CHECK(axon::kernels::parse_kernel_set("neon", parsed));
  CHECK(parsed == axon::kernels::KernelSet::Neon);
  CHECK_FALSE(axon::kernels::parse_kernel_set("fast", parsed));
}

TEST_CASE("a model runs on the table it was asked for", "[kernels][dispatch][model]") {
  const auto path = axon::test::fixture_path("gguf/synthetic-2l.gguf");
  auto scalar = axon::model::LlamaModel::load(
      path, {.max_seq = 32, .max_ctx = 32, .kernels = axon::kernels::KernelSet::Scalar});
  auto best = axon::model::LlamaModel::load(path, {.max_seq = 32, .max_ctx = 32});
  REQUIRE(scalar);
  REQUIRE(best);
  CHECK(&(*scalar)->table() == &axon::kernels::scalar_kernels());
  CHECK(axon::kernels::name((*best)->table()) ==
        axon::kernels::name(**axon::kernels::select(axon::kernels::KernelSet::Auto)));
}
