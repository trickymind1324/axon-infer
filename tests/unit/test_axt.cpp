#include "core/axt.h"

#include "fixture_tensor.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("axt round trip through a file", "[core][axt]") {
  const auto path = std::filesystem::temp_directory_path() / "axon_axt_roundtrip.axt";
  const std::vector<float> values{1.5F, -2.0F, 0.0F, 3.25F, 100.0F, -0.125F};
  const std::uint64_t dims[] = {2, 3};
  const auto written = axon::write_axt_f32(path, dims, values);
  REQUIRE(written);
  CHECK(*written == 4 + 4 + 16 + 4 + 24);

  const auto back = axon::read_axt(path);
  REQUIRE(back);
  CHECK(back->dims == std::vector<std::uint64_t>{2, 3});
  CHECK(back->type == axon::AxtType::F32);
  CHECK(back->as_f32() == values);

  const std::vector<std::int32_t> ids{7, -1, 300};
  const std::uint64_t one[] = {3};
  REQUIRE(axon::write_axt_i32(path, one, ids));
  const auto ints = axon::read_axt(path);
  REQUIRE(ints);
  CHECK(ints->as_i32() == ids);
  std::filesystem::remove(path);
}

TEST_CASE("axt parse errors name the problem", "[core][axt]") {
  const std::vector<std::byte> junk(20, std::byte{0});
  const auto bad_magic = axon::parse_axt(junk);
  REQUIRE_FALSE(bad_magic);
  CHECK_THAT(bad_magic.error().message, ContainsSubstring("bad magic"));

  auto fixture = axon::test::read_fixture("kernels/softmax_5.x.axt");
  fixture.pop_back();
  const auto truncated = axon::parse_axt(fixture);
  REQUIRE_FALSE(truncated);
  CHECK_THAT(truncated.error().message, ContainsSubstring("expected 20 data bytes, found 19"));

  const auto missing = axon::read_axt("/nonexistent/dir/x.axt");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == axon::ErrorCode::Io);
}

TEST_CASE("kernel fixtures load through the shared reader", "[core][axt]") {
  const auto half = axon::test::load_axt("matmul_f16.w");
  CHECK(half.dim(0) == 48);
  CHECK(half.type == axon::AxtType::F16);
  CHECK(half.as_u16().size() == std::size_t{48} * 64);
  CHECK(half.as_f32().size() == std::size_t{48} * 64);
}
