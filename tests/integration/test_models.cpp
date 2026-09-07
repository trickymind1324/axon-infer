// Integration tests against real model files. They run only when AXON_MODELS_DIR points at a
// directory populated by scripts/download_models.sh; otherwise every test is skipped.

#include "core/gguf.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path models_dir() {
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set; run scripts/download_models.sh and export it");
  }
  return env;
}

axon::GGUFFile open_model(const std::string& file) {
  const auto path = models_dir() / file;
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto model = axon::GGUFFile::open(path);
  if (!model) {
    FAIL(model.error().message);
  }
  return std::move(*model);
}

// One "name dtype shape" line per tensor, in file order; the same format the Python
// reference script writes.
std::vector<std::string> tensor_lines(const axon::GGUFFile& file) {
  std::vector<std::string> lines;
  for (const auto& t : file.tensors()) {
    std::string shape;
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
      shape += (d == 0 ? "" : "x") + std::to_string(t.shape[d]);
    }
    lines.push_back(std::string(t.name) + " " + std::string(axon::dtype_name(t.dtype)) + " " +
                    shape);
  }
  return lines;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

}  // namespace

TEST_CASE("SmolLM2-135M Q8_0 tensor table matches the reference listing", "[integration][gguf]") {
  const auto file = open_model("SmolLM2-135M-Instruct-Q8_0.gguf");
  CHECK(file.architecture() == "llama");
  CHECK(file.get_uint("llama.block_count").value() == 30);
  CHECK(file.get_uint("llama.embedding_length").value() == 576);
  CHECK(file.metadata("tokenizer.ggml.tokens")->array_size() == 49152);

  const auto expected = read_lines(std::filesystem::path(AXON_FIXTURE_DIR) / ".." / "expected" /
                                   "smollm2-135m-instruct-q8_0.tensors.txt");
  REQUIRE_FALSE(expected.empty());
  const auto actual = tensor_lines(file);
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    INFO("tensor " << i);
    CHECK(actual[i] == expected[i]);
  }
}

TEST_CASE("SmolLM2-135M F16 and Q4_0 files share the tensor names", "[integration][gguf]") {
  const auto q8 = open_model("SmolLM2-135M-Instruct-Q8_0.gguf");
  for (const char* other : {"SmolLM2-135M-Instruct-f16.gguf", "SmolLM2-135M-Instruct-Q4_0.gguf"}) {
    const auto file = open_model(other);
    REQUIRE(file.tensor_count() == q8.tensor_count());
    for (const auto& t : q8.tensors()) {
      const auto* match = file.tensor(t.name);
      REQUIRE(match != nullptr);
      CHECK(match->shape == t.shape);
      CHECK(match->n_dims == t.n_dims);
    }
  }
}

TEST_CASE("every tensor lies inside the mapped file", "[integration][gguf]") {
  const auto file = open_model("SmolLM2-135M-Instruct-f16.gguf");
  const auto* end = file.tensors().front().data - file.tensors().front().offset -
                    file.data_offset() + file.file_size();
  for (const auto& t : file.tensors()) {
    CHECK(t.data + t.nbytes <= end);
    CHECK(reinterpret_cast<std::uintptr_t>(t.data) % file.alignment() == 0);
  }
}
