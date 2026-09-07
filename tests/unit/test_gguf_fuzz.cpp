#include "core/gguf.h"

#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Small deterministic generator so the corpus of mutations is identical on every run.
class Lcg {
public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return state_ >> 11;
  }
  std::size_t below(std::size_t n) { return n == 0 ? 0 : static_cast<std::size_t>(next() % n); }

private:
  std::uint64_t state_;
};

constexpr std::uint64_t kExtremes[] = {0,
                                       1,
                                       0x7F,
                                       0xFF,
                                       0x7FFF,
                                       0xFFFF,
                                       0x7FFFFFFF,
                                       0xFFFFFFFF,
                                       0x100000000ULL,
                                       0x7FFFFFFFFFFFFFFFULL,
                                       0xFFFFFFFFFFFFFFFFULL};

std::vector<std::byte> mutate(const std::vector<std::byte>& seed, Lcg& rng) {
  std::vector<std::byte> bytes = seed;
  switch (rng.below(6)) {
  case 0: {  // flip random bits
    const std::size_t n = 1 + rng.below(8);
    for (std::size_t i = 0; i < n && !bytes.empty(); ++i) {
      const std::size_t at = rng.below(bytes.size());
      bytes[at] ^= static_cast<std::byte>(1U << rng.below(8));
    }
    break;
  }
  case 1: {  // truncate
    bytes.resize(rng.below(bytes.size() + 1));
    break;
  }
  case 2: {  // overwrite an aligned 8-byte field with an extreme value
    if (bytes.size() >= 8) {
      const std::size_t at = rng.below(bytes.size() - 7);
      const std::uint64_t v = kExtremes[rng.below(std::size(kExtremes))];
      std::memcpy(bytes.data() + at, &v, 8);
    }
    break;
  }
  case 3: {  // overwrite a 4-byte field (types, dim counts) with a small or huge value
    if (bytes.size() >= 4) {
      const std::size_t at = rng.below(bytes.size() - 3);
      const auto v = static_cast<std::uint32_t>(kExtremes[rng.below(std::size(kExtremes))]);
      std::memcpy(bytes.data() + at, &v, 4);
    }
    break;
  }
  case 4: {  // truncate near the header boundaries where most length fields live
    const std::size_t limit = std::min<std::size_t>(bytes.size(), 2048);
    bytes.resize(rng.below(limit + 1));
    break;
  }
  default: {  // append garbage
    const std::size_t n = rng.below(64);
    for (std::size_t i = 0; i < n; ++i) {
      bytes.push_back(static_cast<std::byte>(rng.below(256)));
    }
    break;
  }
  }
  return bytes;
}

void exercise(const axon::GGUFFile& file) {
  // Touch every accessor so a successful parse of a mutated file is also walked.
  volatile std::size_t sink = 0;
  for (const auto& [key, value] : file.metadata_entries()) {
    sink = sink + key.size() + value.array_size();
    if (auto s = value.as_string()) {
      sink = sink + s->size();
    }
  }
  for (const auto& t : file.tensors()) {
    sink = sink + t.name.size() + t.n_elements() % 7;
    if (t.nbytes > 0) {
      sink = sink + static_cast<std::size_t>(t.data[0]) +
             static_cast<std::size_t>(t.data[t.nbytes - 1]);
    }
  }
}

void replay(const char* fixture, int cases) {
  const auto seed = axon::test::read_fixture(fixture);
  Lcg rng(0x5eed1234ULL);
  int ok = 0;
  int failed = 0;
  for (int i = 0; i < cases; ++i) {
    const auto bytes = mutate(seed, rng);
    const auto parsed = axon::GGUFFile::parse(bytes);
    if (parsed) {
      exercise(*parsed);
      ++ok;
    } else {
      REQUIRE_FALSE(parsed.error().message.empty());
      ++failed;
    }
  }
  INFO(fixture << ": " << ok << " parsed, " << failed << " rejected");
  CHECK(ok + failed == cases);
}

}  // namespace

TEST_CASE("corrupted tiny fixtures never crash the parser", "[core][gguf][fuzz]") {
  replay("gguf/tiny.gguf", 5000);
}

TEST_CASE("corrupted synthetic fixtures never crash the parser", "[core][gguf][fuzz]") {
  replay("gguf/synthetic-2l-q4_0.gguf", 2000);
}
