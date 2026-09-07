// Tokenizer exact-match tests against downloaded models (Llama 3.2 and Qwen2.5), run only when
// AXON_MODELS_DIR is set.

#include "axon/tokenizer.h"
#include "core/gguf.h"

#include "tokenizer/nfc.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path models_dir() {
  const char* env = std::getenv("AXON_MODELS_DIR");
  if (env == nullptr || *env == '\0') {
    SKIP("AXON_MODELS_DIR is not set");
  }
  return env;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

void check_model(const char* file, const char* fixture, const char* expected_pre, bool nfc) {
  const auto path = models_dir() / file;
  if (!std::filesystem::exists(path)) {
    SKIP("model file not present: " << path.string());
  }
  auto gguf = axon::GGUFFile::open(path);
  REQUIRE(gguf);
  auto tokenizer = axon::Tokenizer::from_gguf(*gguf);
  if (!tokenizer) {
    FAIL(tokenizer.error().message);
  }
  CHECK((*tokenizer)->pre() == expected_pre);

  const std::filesystem::path fixtures = std::filesystem::path(AXON_FIXTURE_DIR);
  auto lines = read_lines(fixtures / ".." / "tokenizer_corpus.txt");
  {
    std::ifstream in(fixtures / ".." / "tokenizer_corpus.txt", std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    lines.push_back(buffer.str());
  }
  const auto expected = read_lines(fixtures / "tokenizer" / (std::string(fixture) + ".ids.txt"));
  REQUIRE(expected.size() == lines.size());
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::vector<std::int32_t> want;
    std::istringstream stream(expected[i]);
    std::int32_t id = 0;
    while (stream >> id) {
      want.push_back(id);
    }
    const auto ids = (*tokenizer)->encode(lines[i], {.add_bos = false});
    INFO("entry " << i << ": " << lines[i].substr(0, 80));
    if (ids != want) {
      ++mismatches;
    }
    CHECK(ids == want);
    // Qwen normalises to NFC before tokenizing, so its round trip returns the NFC form.
    CHECK((*tokenizer)->decode(ids, false) ==
          (nfc ? axon::unicode::normalize_nfc(lines[i]) : lines[i]));
  }
  CHECK(mismatches == 0);
}

}  // namespace

TEST_CASE("Llama 3.2 tokenizer matches Hugging Face on the corpus", "[integration][tokenizer]") {
  check_model("Llama-3.2-1B-Instruct-Q4_0.gguf", "llama3", "llama-bpe", false);
}

TEST_CASE("Qwen2.5 tokenizer matches Hugging Face on the corpus", "[integration][tokenizer]") {
  check_model("qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2", "qwen2", true);
}
