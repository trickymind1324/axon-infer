#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace axon::test {

inline std::filesystem::path fixture_path(const std::string& relative) {
  return std::filesystem::path(AXON_FIXTURE_DIR) / relative;
}

inline std::vector<std::byte> read_fixture(const std::string& relative) {
  std::ifstream in(fixture_path(relative), std::ios::binary);
  std::vector<char> chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<std::byte> bytes(chars.size());
  for (std::size_t i = 0; i < chars.size(); ++i) {
    bytes[i] = static_cast<std::byte>(chars[i]);
  }
  return bytes;
}

}  // namespace axon::test
