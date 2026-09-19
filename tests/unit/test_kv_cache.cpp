#include "core/gguf.h"

#include "fixtures.h"
#include "model/config.h"
#include "model/kv_cache.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace {

axon::model::LlamaConfig synthetic_config() {
  auto file = axon::GGUFFile::open(axon::test::fixture_path("gguf/synthetic-2l.gguf"));
  REQUIRE(file);
  auto cfg = axon::model::LlamaConfig::from_gguf(*file);
  REQUIRE(cfg);
  return *cfg;
}

}  // namespace

TEST_CASE("kv cache geometry", "[model][kv_cache]") {
  const auto cfg = synthetic_config();
  axon::model::KVCache cache(cfg, 40);
  CHECK(cache.n_layer() == 2);
  CHECK(cache.max_ctx() == 40);
  CHECK(cache.row_floats() == 32);
  CHECK(cache.bytes() == std::size_t{2} * 2 * 40 * 32 * sizeof(float));
  CHECK(cache.pos() == 0);
  CHECK(cache.live_bytes() == 0);
  CHECK(cache.type() == axon::model::CacheType::F32);
  CHECK(cache.matches(cfg));

  auto other = cfg;
  other.n_layer = 3;
  CHECK_FALSE(cache.matches(other));
  other = cfg;
  other.n_head_kv = 4;
  CHECK_FALSE(cache.matches(other));
}

TEST_CASE("kv cache layers and rows do not alias", "[model][kv_cache]") {
  const auto cfg = synthetic_config();
  axon::model::KVCache cache(cfg, 8);
  const std::size_t row = cache.row_floats();
  CHECK(cache.v(0) == cache.k(0) + 8 * row);
  CHECK(cache.k(1) == cache.v(0) + 8 * row);
  CHECK(cache.v(1) == cache.k(1) + 8 * row);
  cache.k(0)[0] = 1.0F;
  cache.v(0)[0] = 2.0F;
  cache.k(1)[0] = 3.0F;
  cache.v(1)[7 * row + row - 1] = 4.0F;
  CHECK(cache.k(0)[0] == 1.0F);
  CHECK(cache.v(0)[0] == 2.0F);
  CHECK(cache.k(1)[0] == 3.0F);
  CHECK(cache.v(1)[8 * row - 1] == 4.0F);
  const auto& const_cache = cache;
  CHECK(const_cache.k(1) == cache.k(1));
}

TEST_CASE("kv cache position bookkeeping", "[model][kv_cache]") {
  const auto cfg = synthetic_config();
  axon::model::KVCache cache(cfg, 16);
  cache.advance(5);
  CHECK(cache.pos() == 5);
  CHECK(cache.live_bytes() == std::size_t{2} * 2 * 5 * 32 * sizeof(float));
  cache.advance(11);
  CHECK(cache.pos() == 16);
  cache.truncate(9);
  CHECK(cache.pos() == 9);
  cache.truncate(9);
  CHECK(cache.pos() == 9);
  cache.clear();
  CHECK(cache.pos() == 0);
}
