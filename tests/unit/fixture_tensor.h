#pragma once

// Test-side helpers over the .axt reader in core: load a kernel or model fixture by name and
// compare float vectors.

#include "core/axt.h"

#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace axon::test {

using AxtType = axon::AxtType;
using FixtureTensor = axon::AxtTensor;

// Loads fixtures/<dir>/<name>.axt.
inline FixtureTensor load_axt(const std::string& name, const std::string& dir = "kernels") {
  auto tensor = read_axt(fixture_path(dir + "/" + name + ".axt"));
  if (!tensor) {
    FAIL("cannot load fixture " << name << ": " << tensor.error().message);
  }
  return std::move(*tensor);
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
    worst = d > worst ? d : worst;
  }
  return worst;
}

}  // namespace axon::test
