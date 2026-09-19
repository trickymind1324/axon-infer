#include "chat/jinja.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>

using axon::chat::List;
using axon::chat::Map;
using axon::chat::Template;
using axon::chat::Value;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string render(const std::string& source, const Map& vars = {}) {
  auto tmpl = Template::parse(source);
  if (!tmpl) {
    FAIL("parse: " << tmpl.error().message);
  }
  auto out = tmpl->render(vars);
  if (!out) {
    FAIL("render: " << out.error().message);
  }
  return *out;
}

std::string parse_error(const std::string& source) {
  auto tmpl = Template::parse(source);
  if (tmpl) {
    FAIL("expected a parse error");
  }
  return tmpl.error().message;
}

std::string render_error(const std::string& source, const Map& vars = {}) {
  auto tmpl = Template::parse(source);
  if (!tmpl) {
    FAIL("parse: " << tmpl.error().message);
  }
  auto out = tmpl->render(vars);
  if (out) {
    FAIL("expected a render error, got '" << *out << "'");
  }
  return out.error().message;
}

Map message(const std::string& role, const std::string& content) {
  return {{"role", Value::string(role)}, {"content", Value::string(content)}};
}

Map chat_vars(List messages, bool generation_prompt = true) {
  return {{"messages", Value::list(std::move(messages))},
          {"add_generation_prompt", Value::boolean(generation_prompt)},
          {"bos_token", Value::string("<s>")},
          {"eos_token", Value::string("</s>")}};
}

}  // namespace

TEST_CASE("text, output and literals", "[chat][jinja]") {
  CHECK(render("hello") == "hello");
  CHECK(render("a{{ 'b' }}c") == "abc");
  CHECK(render("{{ \"q\\\"uote\\n\" }}") == "q\"uote\n");
  CHECK(render("{{ 1 + 2 }}") == "3");
  CHECK(render("{{ 5 - 2 - 1 }}/{{ -3 }}/{{ xs[n - 1] }}",
               {{"xs", Value::list({Value::string("a"), Value::string("b")})},
                {"n", Value::integer(2)}}) == "2/-3/b");
  CHECK(render("{{ true }}/{{ false }}/{{ none }}/{{ missing }}") == "True/False//");
  CHECK(render("{{ 'a' + 'b' + 'c' }}") == "abc");
}

TEST_CASE("names, attributes, subscripts and slices", "[chat][jinja]") {
  const Map vars = {{"m", Value::map(message("user", "hi"))},
                    {"xs", Value::list({Value::integer(1), Value::integer(2), Value::integer(3)})}};
  CHECK(render("{{ m.role }} {{ m['content'] }}", vars) == "user hi");
  CHECK(render("{{ xs[0] }}{{ xs[-1] }}{{ xs[7] }}", vars) == "13");
  CHECK(render("{% for x in xs[1:] %}{{ x }}{% endfor %}", vars) == "23");
  CHECK(render("{% for x in xs[:2] %}{{ x }}{% endfor %}", vars) == "12");
  CHECK(render("{{ m.missing }}|{{ m.role.nothing }}", vars) == "|");
}

TEST_CASE("if, elif, else and boolean logic", "[chat][jinja]") {
  const std::string t = "{% if a == 1 %}one{% elif a == 2 %}two{% else %}other{% endif %}";
  CHECK(render(t, {{"a", Value::integer(1)}}) == "one");
  CHECK(render(t, {{"a", Value::integer(2)}}) == "two");
  CHECK(render(t, {{"a", Value::integer(3)}}) == "other");
  CHECK(render("{% if x and not y or z %}yes{% endif %}",
               {{"x", Value::boolean(true)}, {"y", Value::boolean(false)}}) == "yes");
  CHECK(render("{% if (a != 'b') %}ok{% endif %}", {{"a", Value::string("a")}}) == "ok");
  CHECK(render("{% if missing %}no{% else %}empty is false{% endif %}") == "empty is false");
  CHECK(render("{% if 'role' in m %}has{% endif %}{% if 'x' not in m %} no-x{% endif %}",
               {{"m", Value::map(message("u", "c"))}}) == "has no-x");
}

TEST_CASE("for loops and the loop variable", "[chat][jinja]") {
  const Map vars = {
      {"xs", Value::list({Value::string("a"), Value::string("b"), Value::string("c")})}};
  CHECK(render("{% for x in xs %}{{ loop.index0 }}{{ x }}{% if not loop.last %},{% endif %}"
               "{% endfor %}",
               vars) == "0a,1b,2c");
  CHECK(render("{% for x in xs %}{% if loop.first %}[{% endif %}{{ loop.index }}/{{ loop.length }}"
               "{% if loop.last %}]{% endif %}{% endfor %}",
               vars) == "[1/32/33/3]");
  CHECK(render("{% for x in missing %}never{% endfor %}done") == "done");
}

TEST_CASE("set and scoping", "[chat][jinja]") {
  CHECK(render("{% set x = 'a' + 'b' %}{{ x }}") == "ab");
  CHECK(render("{% set msgs = xs[1:] %}{% for m in msgs %}{{ m }}{% endfor %}",
               {{"xs", Value::list({Value::integer(1), Value::integer(2)})}}) == "2");
  // A set inside a loop body does not leak past the loop.
  CHECK(render("{% for x in xs %}{% set y = x %}{% endfor %}[{{ y }}]",
               {{"xs", Value::list({Value::integer(1)})}}) == "[]");
  // A top-level set shadows a variable for the rest of the template.
  CHECK(render("{% set messages = messages[1:] %}{{ messages | length }}",
               chat_vars({Value::map(message("system", "s")), Value::map(message("user", "u"))})) ==
        "1");
}

TEST_CASE("tests and filters", "[chat][jinja]") {
  const Map vars = {{"s", Value::string("  padded  ")},
                    {"n", Value::none()},
                    {"m", Value::map(message("u", "c"))}};
  CHECK(render("{% if s is defined %}d{% endif %}{% if q is not defined %}u{% endif %}", vars) ==
        "du");
  CHECK(render("{% if n is none %}n{% endif %}{% if s is not none %}s{% endif %}", vars) == "ns");
  CHECK(render("{% if m is mapping %}map{% endif %}{% if s is iterable %} it{% endif %}", vars) ==
        "map it");
  CHECK(render("[{{ s | trim }}]{{ s | length }}{{ m | length }}", vars) == "[padded]102");
  CHECK(render("{{ 'x' + s | trim + 'y' }}", vars) == "xpaddedy");  // filter binds tighter than +
  CHECK(render("{% if strftime_now is defined %}yes{% endif %}") == "yes");
  CHECK(render("{{ strftime_now('%Y') }}").size() == 4);
}

TEST_CASE("whitespace control", "[chat][jinja]") {
  CHECK(render("a  {{- 'b' -}}  c") == "abc");
  CHECK(render("a\n  {%- if true %}b{% endif -%}\n\nc") == "ab"
                                                           "c");
  // trim_blocks: the newline right after a block tag disappears. lstrip_blocks: indentation
  // before a block tag on its own line disappears.
  CHECK(render("{% if true %}\nline\n{% endif %}\n") == "line\n");
  CHECK(render("x\n    {% if true %}y{% endif %}\nz") == "x\nyz");
  CHECK(render("x\n    {% if true %}y\n    {% endif %}\nz") == "x\ny\nz");
  CHECK(render("  {% if true %}y{% endif %}z") == "yz");
  // Output tags keep their surrounding whitespace.
  CHECK(render("a\n{{ 'b' }}\nc") == "a\nb\nc");
  CHECK(render("{# a comment #}text{#- another -#}   more") == "textmore");
}

TEST_CASE("errors name the construct", "[chat][jinja]") {
  CHECK_THAT(parse_error("{% macro x() %}{% endmacro %}"),
             ContainsSubstring("unsupported tag 'macro'"));
  CHECK_THAT(parse_error("{% if x %}open"), ContainsSubstring("missing 'elif'"));
  CHECK_THAT(parse_error("{{ x is weird }}"), ContainsSubstring("unsupported test 'weird'"));
  CHECK_THAT(parse_error("{{ 'unterminated }}"), ContainsSubstring("unterminated"));
  CHECK_THAT(parse_error("{{ x "), ContainsSubstring("unterminated"));
  CHECK_THAT(render_error("{{ m | tojson }}", {{"m", Value::map({})}}),
             ContainsSubstring("unsupported filter 'tojson'"));
  CHECK_THAT(render_error("{{ raise_exception('boom') }}"), ContainsSubstring("boom"));
  CHECK_THAT(render_error("{{ 1 + 'a' }}"), ContainsSubstring("cannot add"));
  // Unreached branches with unsupported filters parse fine and never fail.
  CHECK(render("{% if tools %}{{ tools | tojson(indent=4) }}{% endif %}ok") == "ok");
}

TEST_CASE("a ChatML template renders end to end", "[chat][jinja]") {
  const std::string chatml =
      "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
      "{{ '<|im_start|>system\nYou are helpful<|im_end|>\n' }}{% endif %}"
      "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
      "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
  CHECK(render(chatml, chat_vars({Value::map(message("user", "Hi"))})) ==
        "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n");
}
