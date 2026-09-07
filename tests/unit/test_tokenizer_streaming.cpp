#include "tokenizer/unicode.h"
#include "tokenizer_corpus.h"

#include <catch2/catch_test_macros.hpp>

namespace {

void check_streaming(const axon::Tokenizer& tok, const axon::test::Corpus& corpus) {
  for (std::size_t i = 0; i < corpus.lines.size(); ++i) {
    axon::StreamingDecoder decoder(tok);
    std::string streamed;
    for (const std::int32_t id : corpus.expected[i]) {
      const std::string chunk = decoder.push(id);
      REQUIRE(axon::unicode::is_valid(chunk));  // never a partial sequence
      streamed += chunk;
    }
    streamed += decoder.flush();
    INFO("entry " << i);
    CHECK(streamed == tok.decode(corpus.expected[i]));
  }
}

}  // namespace

TEST_CASE("streaming decode never emits partial UTF-8 (SmolLM2)", "[tokenizer][streaming]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/smollm2.tokenizer.gguf"));
  check_streaming(*tok, axon::test::load_corpus("smollm2"));
}

TEST_CASE("streaming decode never emits partial UTF-8 (TinyLlama)", "[tokenizer][streaming]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/tinyllama.tokenizer.gguf"));
  check_streaming(*tok, axon::test::load_corpus("tinyllama"));
}

TEST_CASE("streaming decoder holds back an incomplete emoji", "[tokenizer][streaming]") {
  const auto tok =
      axon::test::open_tokenizer(axon::test::fixture_path("tokenizer/tinyllama.tokenizer.gguf"));
  const auto all =
      tok->encode("\xF0\x9F\xA6\x99", {.add_bos = false});  // prefix + four byte tokens
  REQUIRE(all.size() == 5);
  const std::vector<std::int32_t> ids(all.begin() + 1, all.end());
  axon::StreamingDecoder decoder(*tok);
  CHECK(decoder.push(ids[0]).empty());
  CHECK(decoder.push(ids[1]).empty());
  CHECK(decoder.push(ids[2]).empty());
  CHECK(decoder.push(ids[3]) == "\xF0\x9F\xA6\x99");
  CHECK(decoder.flush().empty());

  // A truncated sequence becomes U+FFFD at flush.
  axon::StreamingDecoder truncated(*tok);
  truncated.push(ids[0]);
  truncated.push(ids[1]);
  CHECK(truncated.flush() == "\xEF\xBF\xBD");
}
