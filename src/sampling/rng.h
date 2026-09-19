#pragma once

// xoshiro256** seeded through SplitMix64. Written here rather than taken from <random> because
// the standard distributions are not bit-specified across libstdc++ and libc++, and a seeded
// generation must reproduce the same tokens on every platform.

#include <cstdint>

namespace axon {

class Rng {
public:
  explicit Rng(std::uint64_t seed = 0) { this->seed(seed); }

  void seed(std::uint64_t seed) {
    std::uint64_t x = seed;
    for (auto& word : state_) {
      x += 0x9E3779B97F4A7C15ULL;
      std::uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      word = z ^ (z >> 31);
    }
  }

  std::uint64_t next() {
    const std::uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const std::uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

  // Uniform in [0, 1) from the top 53 bits, exactly representable as a double.
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  std::uint64_t state_[4]{};
};

}  // namespace axon
