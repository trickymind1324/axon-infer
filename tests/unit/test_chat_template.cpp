#include "axon/tokenizer.h"
#include "core/gguf.h"

#include "chat/template.h"
#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using axon::chat::ChatTemplate;
using axon::chat::Family;
using axon::chat::Message;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string read_text(const std::string& relative) {
  std::ifstream in(axon::test::fixture_path(relative), std::ios::binary);
  REQUIRE(in);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// The same three conversations gen_chat_fixtures.py renders through Hugging Face.
struct Conversation {
  std::vector<Message> messages;
  bool add_generation_prompt;
};

std::vector<Conversation> conversations() {
  return {
      {{{"system", "You are a terse assistant."}, {"user", "Name three primes."}}, true},
      {{{"user", "Hello there!"},
        {"assistant", "Hi. How can I help?"},
        {"user", "What's 2 + 2?"},
        {"assistant", "4."},
        {"user", "And times 3?  "},
        {"assistant", "12."}},
       false},
      {{{"user", "  Übersetze: naïve café\n\nzweite Zeile "}}, true},
  };
}

ChatTemplate load_family(const std::string& family, Family fallback) {
  const auto source = read_text("chat/" + family + ".template.jinja");
  std::istringstream tokens(read_text("chat/" + family + ".tokens.txt"));
  std::string bos;
  std::string eos;
  std::getline(tokens, bos);
  std::getline(tokens, eos);
  return ChatTemplate::from_source(source, fallback, bos, eos);
}

}  // namespace

TEST_CASE("the four model templates render exactly like Hugging Face", "[chat][template]") {
  struct Case {
    const char* family;
    Family fallback;
  };
  const Case cases[] = {{.family = "smollm2", .fallback = Family::ChatML},
                        {.family = "tinyllama", .fallback = Family::Zephyr},
                        {.family = "llama3", .fallback = Family::Llama3},
                        {.family = "qwen2", .fallback = Family::ChatML}};
  const auto convs = conversations();
  for (const auto& c : cases) {
    const auto tmpl = load_family(c.family, c.fallback);
    INFO(c.family);
    REQUIRE(tmpl.has_jinja());
    CHECK(tmpl.source() == "jinja");
    for (std::size_t i = 0; i < convs.size(); ++i) {
      const auto expected =
          read_text(std::string("chat/") + c.family + "." + std::to_string(i) + ".txt");
      const auto got = tmpl.render_jinja(convs[i].messages, convs[i].add_generation_prompt,
                                         {.date_string = "26 Jul 2024"});
      INFO("conversation " << i);
      REQUIRE(got);
      CHECK(*got == expected);
    }
  }
}

TEST_CASE("templates from tokenizer-only GGUF files parse and detect their family",
          "[chat][template]") {
  for (const char* name : {"smollm2", "tinyllama"}) {
    auto file = axon::GGUFFile::open(
        axon::test::fixture_path(std::string("tokenizer/") + name + ".tokenizer.gguf"));
    REQUIRE(file);
    auto tokenizer = axon::Tokenizer::from_gguf(*file);
    REQUIRE(tokenizer);
    const auto tmpl = ChatTemplate::from_tokenizer(**tokenizer);
    INFO(name);
    CHECK(tmpl.has_jinja());
    CHECK(tmpl.fallback() == (std::string(name) == "smollm2" ? Family::ChatML : Family::Zephyr));
    const std::vector<Message> messages{{"user", "Hi"}};
    const auto rendered = tmpl.render(messages, true);
    CHECK_THAT(rendered, ContainsSubstring("Hi"));
  }
}

TEST_CASE("fallback families render their shapes", "[chat][template]") {
  const std::vector<Message> messages{{"system", "Be brief."}, {"user", "Hi"}};
  const auto chatml = ChatTemplate::from_source("", Family::ChatML, "<s>", "</s>");
  CHECK(chatml.source() == "chatml");
  CHECK(chatml.render(messages, true) ==
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n");
  const auto llama3 =
      ChatTemplate::from_source("", Family::Llama3, "<|begin_of_text|>", "<|eot_id|>");
  CHECK(llama3.render(messages, true) ==
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n");
  const auto zephyr = ChatTemplate::from_source("", Family::Zephyr, "<s>", "</s>");
  CHECK(zephyr.render(messages, true) ==
        "<|system|>\nBe brief.</s>\n<|user|>\nHi</s>\n<|assistant|>\n");
  const auto plain = ChatTemplate::from_source("", Family::Plain, "", "");
  CHECK(plain.render(messages, true) == "System: Be brief.\nUser: Hi\nAssistant:");
  CHECK(plain.render(messages, false) == "System: Be brief.\nUser: Hi\n");
}

TEST_CASE("an unsupported template falls back and says which family", "[chat][template]") {
  const auto tmpl =
      ChatTemplate::from_source("{% macro m() %}x{% endmacro %}", Family::ChatML, "<s>", "</s>");
  CHECK_FALSE(tmpl.has_jinja());
  CHECK(tmpl.source() == "chatml");
  const std::vector<Message> messages{{"user", "Hi"}};
  CHECK(tmpl.render(messages, true) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

  // A runtime failure falls back for that render only.
  const auto raising =
      ChatTemplate::from_source("{{ raise_exception('no') }}", Family::Zephyr, "<s>", "</s>");
  CHECK(raising.has_jinja());
  CHECK(raising.render(messages, false) == "<|user|>\nHi</s>\n");
  CHECK_FALSE(raising.render_jinja(messages, false));
}
