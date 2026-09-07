#pragma once

#include "axon/tokenizer.h"
#include "core/gguf.h"

#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace axon::test {

struct Corpus {
  std::vector<std::string> lines;  // corpus lines, then the whole document as the last entry
  std::vector<std::vector<std::int32_t>> expected;
};

inline std::vector<std::string> read_text_lines(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

inline Corpus load_corpus(const std::string& ids_name) {
  Corpus corpus;
  const auto corpus_path = fixture_path("../tokenizer_corpus.txt");
  corpus.lines = read_text_lines(corpus_path);
  std::ifstream in(corpus_path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  corpus.lines.push_back(buffer.str());
  for (const auto& line : read_text_lines(fixture_path("tokenizer/" + ids_name + ".ids.txt"))) {
    std::vector<std::int32_t> ids;
    std::istringstream stream(line);
    std::int32_t id = 0;
    while (stream >> id) {
      ids.push_back(id);
    }
    corpus.expected.push_back(std::move(ids));
  }
  REQUIRE(corpus.expected.size() == corpus.lines.size());
  return corpus;
}

inline std::unique_ptr<Tokenizer> open_tokenizer(const std::filesystem::path& path) {
  auto file = GGUFFile::open(path);
  if (!file) {
    FAIL("cannot open " << path.string() << ": " << file.error().message);
  }
  auto tokenizer = Tokenizer::from_gguf(*file);
  if (!tokenizer) {
    FAIL("cannot build tokenizer: " << tokenizer.error().message);
  }
  return std::move(*tokenizer);
}

// Encode every corpus entry without BOS and compare ids exactly; also round-trip through decode.
inline void check_corpus(const Tokenizer& tokenizer, const Corpus& corpus, bool check_round_trip) {
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < corpus.lines.size(); ++i) {
    const auto ids = tokenizer.encode(corpus.lines[i], {.add_bos = false});
    INFO("entry " << i << ": " << corpus.lines[i].substr(0, 80));
    if (ids != corpus.expected[i]) {
      ++mismatches;
    }
    CHECK(ids == corpus.expected[i]);
    if (check_round_trip) {
      CHECK(tokenizer.decode(ids, false) == corpus.lines[i]);  // keep specials for the round trip
    }
  }
  CHECK(mismatches == 0);
}

}  // namespace axon::test
