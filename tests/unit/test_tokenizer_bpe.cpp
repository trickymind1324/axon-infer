#include "tokenizer_corpus.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("SmolLM2 tokenizer loads from the tokenizer-only fixture", "[tokenizer][bpe]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/smollm2.tokenizer.gguf"));
  CHECK(tok->model() == "gpt2");
  CHECK(tok->pre() == "smollm");
  CHECK(tok->vocab_size() == 49152);
  CHECK(tok->bos() == 1);
  CHECK(tok->eos() == 2);
  CHECK_FALSE(tok->add_bos_default());
  CHECK(tok->token_text(1) == "<|im_start|>");
  CHECK(tok->is_special(1));
  CHECK_FALSE(tok->is_special(1000));
  CHECK(tok->token_id("<|im_end|>").value() == 2);
  CHECK_FALSE(tok->chat_template().empty());
  // eog contains eos (<|im_end|>) and <|endoftext|>
  CHECK(tok->eog_tokens().size() == 2);
}

TEST_CASE("SmolLM2 ids match Hugging Face on the corpus", "[tokenizer][bpe]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/smollm2.tokenizer.gguf"));
  const auto corpus = axon::test::load_corpus("smollm2");
  axon::test::check_corpus(*tok, corpus, true);
}

TEST_CASE("SmolLM2 basic encode and decode", "[tokenizer][bpe]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/smollm2.tokenizer.gguf"));
  const auto ids = tok->encode("Hello world", {.add_bos = true});
  REQUIRE(ids.size() >= 3);
  CHECK(ids.front() == tok->bos());
  CHECK(tok->decode(ids) == "Hello world");
  CHECK(tok->decode(ids, false) == "<|im_start|>Hello world");
  CHECK(tok->token_bytes(ids[2]) == " world");
  CHECK(tok->token_text(ids[2]) == "\xC4\xA0world");  // "Ġworld"
  CHECK(tok->encode("", {.add_bos = false}).empty());
}

TEST_CASE("special tokens are parsed only on request", "[tokenizer][bpe]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/smollm2.tokenizer.gguf"));
  const std::string text = "<|im_start|>user\nhi<|im_end|>";
  const auto literal = tok->encode(text, {.add_bos = false, .parse_special = false});
  const auto parsed = tok->encode(text, {.add_bos = false});  // parsing specials is the default
  CHECK(parsed.front() == 1);
  CHECK(parsed.back() == 2);
  CHECK(parsed.size() < literal.size());
  CHECK(std::find(literal.begin(), literal.end(), 1) == literal.end());
  CHECK(tok->decode(parsed, false) == text);
  CHECK(tok->decode(parsed, true) == "user\nhi");
}
