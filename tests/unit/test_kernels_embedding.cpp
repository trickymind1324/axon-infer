#include "core/dtype.h"

#include "fixture_tensor.h"
#include "kernels/kernels.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <vector>

TEST_CASE("embedding_row gathers F32, F16 and BF16 rows", "[kernels][embedding]") {
  const auto& k = axon::kernels::kernels();
  struct Case {
    const char* name;
    axon::DType type;
  };
  for (const auto& c : {Case{.name = "embedding_f32", .type = axon::DType::F32},
                        Case{.name = "embedding_f16", .type = axon::DType::F16},
                        Case{.name = "embedding_bf16", .type = axon::DType::BF16}}) {
    const std::string base = c.name;
    const auto table = axon::test::load_axt(base + ".table");
    const auto index = axon::test::load_axt(base + ".index").as_i32();
    const auto rows = axon::test::load_axt(base + ".rows");
    const auto expected = rows.as_f32();
    const std::size_t n = table.dim(1);
    REQUIRE(rows.dim(1) == n);
    REQUIRE(rows.dim(0) == index.size());
    std::vector<float> got(index.size() * n);
    for (std::size_t i = 0; i < index.size(); ++i) {
      k.embedding_row(c.type, table.data.data(), n, index[i], got.data() + i * n);
    }
    INFO(c.name);
    // Every conversion is exact, so the gathered rows must be bit-identical.
    CHECK(axon::test::max_abs_diff(got, expected) == 0.0F);
  }
}
