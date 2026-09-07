#include "tokenizer_corpus.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("TinyLlama tokenizer loads from the tokenizer-only fixture", "[tokenizer][spm]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/tinyllama.tokenizer.gguf"));
  CHECK(tok->model() == "llama");
  CHECK(tok->vocab_size() == 32000);
  CHECK(tok->bos() == 1);
  CHECK(tok->eos() == 2);
  CHECK(tok->add_bos_default());
  CHECK(tok->token_text(1) == "<s>");
  CHECK(tok->token_type(3) == axon::TokenType::Byte);
  CHECK(tok->token_bytes(3) == std::string(1, '\0'));
}

TEST_CASE("TinyLlama ids match Hugging Face on the corpus", "[tokenizer][spm]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/tinyllama.tokenizer.gguf"));
  const auto corpus = axon::test::load_corpus("tinyllama");
  axon::test::check_corpus(*tok, corpus, false);
}

TEST_CASE("TinyLlama decode drops the synthetic leading space", "[tokenizer][spm]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/tinyllama.tokenizer.gguf"));
  const auto ids = tok->encode("Hello world");
  REQUIRE(ids.size() >= 3);
  CHECK(ids.front() == 1);
  CHECK(tok->decode(ids) == "Hello world");
  // A single leading space is absorbed by the meta-space prefix, as in Hugging Face.
  CHECK(tok->decode(tok->encode(" leading", {.add_bos = false})) == "leading");
  CHECK(tok->decode(tok->encode("  two", {.add_bos = false})) == " two");
  // Byte fallback for a character outside the vocabulary.
  const auto emoji = tok->encode("\xF0\x9F\xA6\x99", {.add_bos = false});  // 🦙: prefix + 4 bytes
  CHECK(emoji.size() == 5);
  CHECK(tok->decode(emoji) == "\xF0\x9F\xA6\x99");
}
