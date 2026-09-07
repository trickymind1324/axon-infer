#include "fixtures.h"
#include "tokenizer/pretokenizer.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(axon::PreType type, std::string_view text) {
  std::vector<axon::ByteRange> ranges;
  axon::pretokenize(type, text, ranges);
  std::vector<std::string> out;
  for (const auto& [b, e] : ranges) {
    out.emplace_back(text.substr(b, e - b));
  }
  return out;
}

std::string from_hex(const std::string& hex) {
  std::string out;
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

// Compares against the chunk fixture written by gen_tokenizer_fixtures.py (HF pre_tokenize_str).
void check_against_fixture(axon::PreType type, const char* name) {
  const auto corpus_path = axon::test::fixture_path("../tokenizer_corpus.txt");
  std::ifstream in(corpus_path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string corpus = buffer.str();
  std::vector<std::string> lines = read_lines(corpus_path.string());
  lines.push_back(corpus);  // the whole document, newlines included

  const auto expected = read_lines(
      axon::test::fixture_path(std::string("tokenizer/") + name + ".pretok.txt").string());
  REQUIRE(expected.size() == lines.size());
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::vector<std::string> want;
    std::istringstream chunks(expected[i]);
    std::string hex;
    while (chunks >> hex) {
      want.push_back(from_hex(hex));
    }
    INFO("line " << i << ": " << lines[i]);
    CHECK(split(type, lines[i]) == want);
  }
}

}  // namespace

TEST_CASE("gpt2 pre-tokenizer on crafted strings", "[tokenizer][pretok]") {
  using V = std::vector<std::string>;
  CHECK(split(axon::PreType::Gpt2, "Hello world") == V{"Hello", " world"});
  CHECK(split(axon::PreType::Gpt2, "I can't") == V{"I", " can", "'t"});
  CHECK(split(axon::PreType::Gpt2, "I CAN'T") == V{"I", " CAN", "'", "T"});
  CHECK(split(axon::PreType::Gpt2, "  abc") == V{" ", " abc"});
  CHECK(split(axon::PreType::Gpt2, "abc  ") == V{"abc", "  "});
  CHECK(split(axon::PreType::Gpt2, "a\tb") == V{"a", "\t", "b"});
  CHECK(split(axon::PreType::Gpt2, "x 1234 y") == V{"x", " 1234", " y"});
  CHECK(split(axon::PreType::Gpt2, "a\n\nb") ==
        V{"a", "\n", "\n", "b"});  // \s+(?!\S) leaves the last newline
  CHECK(split(axon::PreType::Gpt2, "a \n b") == V{"a", " \n", " b"});
  CHECK(split(axon::PreType::Gpt2, "!!! ?") == V{"!!!", " ?"});
  CHECK(split(axon::PreType::Gpt2, "") == V{});
  CHECK(split(axon::PreType::Gpt2, " ") == V{" "});
  CHECK(split(axon::PreType::Gpt2, "中文 text") == V{"中文", " text"});
}

TEST_CASE("llama3 pre-tokenizer on crafted strings", "[tokenizer][pretok]") {
  using V = std::vector<std::string>;
  CHECK(split(axon::PreType::Llama3, "I CAN'T") == V{"I", " CAN", "'T"});
  CHECK(split(axon::PreType::Llama3, "x 1234 y") == V{"x", " ", "123", "4", " y"});
  CHECK(split(axon::PreType::Llama3, "a\n\nb") == V{"a", "\n\n", "b"});
  CHECK(split(axon::PreType::Llama3, "a \n b") == V{"a", " \n", " b"});
  CHECK(split(axon::PreType::Llama3, "a  \n\n  b") == V{"a", "  \n\n", " ", " b"});
  CHECK(split(axon::PreType::Llama3, "!!!\n\nx") == V{"!!!\n\n", "x"});
  CHECK(split(axon::PreType::Llama3, "\tabc") == V{"\tabc"});
  CHECK(split(axon::PreType::Llama3, "\nabc") == V{"\n", "abc"});
  CHECK(split(axon::PreType::Qwen2, "x 1234 y") == V{"x", " ", "1", "2", "3", "4", " y"});
}

TEST_CASE("pre-tokenizer names", "[tokenizer][pretok]") {
  CHECK(axon::pre_type_from_name("smollm").value() == axon::PreType::Gpt2);
  CHECK(axon::pre_type_from_name("llama-bpe").value() == axon::PreType::Llama3);
  CHECK(axon::pre_type_from_name("qwen2").value() == axon::PreType::Qwen2);
  const auto bad = axon::pre_type_from_name("deepseek-llm");
  REQUIRE_FALSE(bad);
  CHECK(bad.error().code == axon::ErrorCode::Unsupported);
  CHECK(axon::pre_type_needs_nfc(axon::PreType::Qwen2));
  CHECK_FALSE(axon::pre_type_needs_nfc(axon::PreType::Gpt2));
}

TEST_CASE("gpt2 pre-tokenizer matches Hugging Face on the corpus", "[tokenizer][pretok]") {
  check_against_fixture(axon::PreType::Gpt2, "smollm2");
}

TEST_CASE("llama3 pre-tokenizer matches Hugging Face on the corpus", "[tokenizer][pretok]") {
  check_against_fixture(axon::PreType::Llama3, "llama3");
}

TEST_CASE("qwen2 pre-tokenizer matches Hugging Face on the corpus", "[tokenizer][pretok]") {
  check_against_fixture(axon::PreType::Qwen2, "qwen2");
}
