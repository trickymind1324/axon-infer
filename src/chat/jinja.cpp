#include "chat/jinja.h"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <optional>

namespace axon::chat {

// --- values -------------------------------------------------------------------------------

Value Value::none() {
  Value v;
  v.kind_ = Kind::None;
  return v;
}
Value Value::boolean(bool b) {
  Value v;
  v.kind_ = Kind::Bool;
  v.bool_ = b;
  return v;
}
Value Value::integer(std::int64_t i) {
  Value v;
  v.kind_ = Kind::Int;
  v.int_ = i;
  return v;
}
Value Value::string(std::string s) {
  Value v;
  v.kind_ = Kind::String;
  v.string_ = std::move(s);
  return v;
}
Value Value::list(List items) {
  Value v;
  v.kind_ = Kind::List;
  v.list_ = std::move(items);
  return v;
}
Value Value::map(Map entries) {
  Value v;
  v.kind_ = Kind::Map;
  v.map_ = std::move(entries);
  return v;
}

bool Value::truthy() const {
  switch (kind_) {
  case Kind::Undefined:
  case Kind::None:
    return false;
  case Kind::Bool:
    return bool_;
  case Kind::Int:
    return int_ != 0;
  case Kind::String:
    return !string_.empty();
  case Kind::List:
    return !list_.empty();
  case Kind::Map:
    return !map_.empty();
  }
  return false;
}

const Value* Value::get(std::string_view key) const {
  if (kind_ != Kind::Map) {
    return nullptr;
  }
  for (const auto& [k, v] : map_) {
    if (k == key) {
      return &v;
    }
  }
  return nullptr;
}

bool Value::equals(const Value& other) const {
  if (kind_ != other.kind_) {
    return false;
  }
  switch (kind_) {
  case Kind::Undefined:
  case Kind::None:
    return true;
  case Kind::Bool:
    return bool_ == other.bool_;
  case Kind::Int:
    return int_ == other.int_;
  case Kind::String:
    return string_ == other.string_;
  case Kind::List:
    if (list_.size() != other.list_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < list_.size(); ++i) {
      if (!list_[i].equals(other.list_[i])) {
        return false;
      }
    }
    return true;
  case Kind::Map:
    if (map_.size() != other.map_.size()) {
      return false;
    }
    for (const auto& [k, v] : map_) {
      const Value* o = other.get(k);
      if (o == nullptr || !v.equals(*o)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string Value::kind_name() const {
  switch (kind_) {
  case Kind::Undefined:
    return "undefined";
  case Kind::None:
    return "none";
  case Kind::Bool:
    return "bool";
  case Kind::Int:
    return "int";
  case Kind::String:
    return "string";
  case Kind::List:
    return "list";
  case Kind::Map:
    return "map";
  }
  return "?";
}

// --- syntax tree --------------------------------------------------------------------------

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
  enum class Kind {
    Literal,
    Name,
    Attr,   // a.b or a['b']: object in `left`, key in `name`
    Index,  // a[expr]
    Slice,  // a[start:stop]
    Add,
    Sub,
    Eq,
    Ne,
    And,
    Or,
    Not,
    In,
    NotIn,
    Test,    // left is <name>, negated when `negate`
    Filter,  // left | name(args)
    Call,    // name(args)
  };
  Kind kind = Kind::Literal;
  Value literal;
  std::string name;
  bool negate = false;
  ExprPtr left;
  ExprPtr right;
  ExprPtr third;
  std::vector<ExprPtr> args;
};

struct Node {
  enum class Kind { Text, Output, If, For, Set };
  Kind kind = Kind::Text;
  std::string text;
  ExprPtr expr;
  std::string name;
  // If: one (condition, body) per if/elif branch, else_body for the tail.
  std::vector<std::pair<ExprPtr, std::vector<std::unique_ptr<Node>>>> branches;
  std::vector<std::unique_ptr<Node>> else_body;
  std::vector<std::unique_ptr<Node>> body;  // For
};

namespace {

using Body = std::vector<std::unique_ptr<Node>>;

// --- template lexer -----------------------------------------------------------------------

struct Chunk {
  enum class Kind { Text, Output, Block, Comment };
  Kind kind;
  std::string content;
  bool strip_before = false;  // opener had `-`
  bool strip_after = false;   // closer had `-`
  std::size_t line = 1;
};

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

Expected<std::vector<Chunk>> lex_template(std::string_view src) {
  std::vector<Chunk> chunks;
  std::size_t i = 0;
  std::size_t line = 1;
  std::string text;
  const auto flush_text = [&] {
    if (!text.empty()) {
      chunks.push_back({.kind = Chunk::Kind::Text, .content = text, .line = line});
      text.clear();
    }
  };
  while (i < src.size()) {
    if (src[i] == '{' && i + 1 < src.size() &&
        (src[i + 1] == '{' || src[i + 1] == '%' || src[i + 1] == '#')) {
      const char open = src[i + 1];
      const Chunk::Kind kind = open == '{'   ? Chunk::Kind::Output
                               : open == '%' ? Chunk::Kind::Block
                                             : Chunk::Kind::Comment;
      const std::string_view close = open == '{' ? "}}" : open == '%' ? "%}" : "#}";
      flush_text();
      std::size_t j = i + 2;
      bool strip_before = false;
      if (j < src.size() && src[j] == '-') {
        strip_before = true;
        ++j;
      }
      const std::size_t start_line = line;
      std::size_t end = std::string_view::npos;
      // Find the closer, skipping over string literals inside expressions.
      char quote = 0;
      for (std::size_t k = j; k + 1 < src.size(); ++k) {
        if (src[k] == '\n') {
          ++line;
        }
        if (quote != 0) {
          if (src[k] == '\\') {
            ++k;
          } else if (src[k] == quote) {
            quote = 0;
          }
          continue;
        }
        if (kind != Chunk::Kind::Comment && (src[k] == '\'' || src[k] == '"')) {
          quote = src[k];
          continue;
        }
        if (src.substr(k, 2) == close) {
          end = k;
          break;
        }
      }
      if (end == std::string_view::npos) {
        return make_error(ErrorCode::Parse,
                          fmt::format("jinja: line {}: unterminated '{{{}'", start_line, open));
      }
      bool strip_after = false;
      std::size_t content_end = end;
      if (content_end > j && src[content_end - 1] == '-') {
        strip_after = true;
        --content_end;
      }
      chunks.push_back({.kind = kind,
                        .content = std::string(src.substr(j, content_end - j)),
                        .strip_before = strip_before,
                        .strip_after = strip_after,
                        .line = start_line});
      i = end + 2;
      continue;
    }
    if (src[i] == '\n') {
      ++line;
    }
    text.push_back(src[i]);
    ++i;
  }
  flush_text();

  // Whitespace control. `-` strips all adjacent whitespace. Otherwise, for block and comment
  // tags, lstrip_blocks removes spaces and tabs from the start of the tag's line and
  // trim_blocks removes the first newline after the tag.
  for (std::size_t c = 0; c < chunks.size(); ++c) {
    Chunk& chunk = chunks[c];
    if (chunk.kind == Chunk::Kind::Text) {
      continue;
    }
    const bool block_like = chunk.kind != Chunk::Kind::Output;
    if (c > 0 && chunks[c - 1].kind == Chunk::Kind::Text) {
      std::string& prev = chunks[c - 1].content;
      if (chunk.strip_before) {
        while (!prev.empty() && is_space(prev.back())) {
          prev.pop_back();
        }
      } else if (block_like) {
        std::size_t k = prev.size();
        while (k > 0 && (prev[k - 1] == ' ' || prev[k - 1] == '\t')) {
          --k;
        }
        const bool line_start = k == 0 ? (c == 1) : prev[k - 1] == '\n';
        if (line_start) {
          prev.resize(k);
        }
      }
    } else if (block_like && chunk.strip_before) {
      // nothing before us to strip
    }
    if (c + 1 < chunks.size() && chunks[c + 1].kind == Chunk::Kind::Text) {
      std::string& next = chunks[c + 1].content;
      if (chunk.strip_after) {
        std::size_t k = 0;
        while (k < next.size() && is_space(next[k])) {
          ++k;
        }
        next.erase(0, k);
      } else if (block_like) {
        if (next.starts_with("\r\n")) {
          next.erase(0, 2);
        } else if (next.starts_with('\n')) {
          next.erase(0, 1);
        }
      }
    }
  }
  return chunks;
}

// --- expression lexer ---------------------------------------------------------------------

struct Token {
  enum class Kind { Name, String, Int, Op, End };
  Kind kind = Kind::End;
  std::string text;
};

Expected<std::vector<Token>> lex_expression(std::string_view src, std::size_t line) {
  std::vector<Token> out;
  std::size_t i = 0;
  while (i < src.size()) {
    const char c = src[i];
    if (is_space(c)) {
      ++i;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_') {
      std::size_t j = i;
      while (j < src.size() &&
             (std::isalnum(static_cast<unsigned char>(src[j])) != 0 || src[j] == '_')) {
        ++j;
      }
      out.push_back({.kind = Token::Kind::Name, .text = std::string(src.substr(i, j - i))});
      i = j;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
      std::size_t j = i;
      while (j < src.size() && std::isdigit(static_cast<unsigned char>(src[j])) != 0) {
        ++j;
      }
      out.push_back({.kind = Token::Kind::Int, .text = std::string(src.substr(i, j - i))});
      i = j;
      continue;
    }
    if (c == '\'' || c == '"') {
      std::string value;
      std::size_t j = i + 1;
      bool closed = false;
      while (j < src.size()) {
        if (src[j] == '\\' && j + 1 < src.size()) {
          const char e = src[j + 1];
          value.push_back(e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e);
          j += 2;
          continue;
        }
        if (src[j] == c) {
          closed = true;
          break;
        }
        value.push_back(src[j]);
        ++j;
      }
      if (!closed) {
        return make_error(ErrorCode::Parse,
                          fmt::format("jinja: line {}: unterminated string", line));
      }
      out.push_back({.kind = Token::Kind::String, .text = value});
      i = j + 1;
      continue;
    }
    static constexpr std::string_view kTwo[] = {"==", "!="};
    bool matched = false;
    for (const auto op : kTwo) {
      if (src.substr(i, 2) == op) {
        out.push_back({.kind = Token::Kind::Op, .text = std::string(op)});
        i += 2;
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }
    if (std::string_view("|.[]():,+-=").find(c) != std::string_view::npos) {
      out.push_back({.kind = Token::Kind::Op, .text = std::string(1, c)});
      ++i;
      continue;
    }
    return make_error(ErrorCode::Parse,
                      fmt::format("jinja: line {}: unexpected character '{}'", line, c));
  }
  out.push_back({.kind = Token::Kind::End});
  return out;
}

// --- expression parser --------------------------------------------------------------------

class ExprParser {
public:
  ExprParser(std::vector<Token> tokens, std::size_t line)
      : tokens_(std::move(tokens)), line_(line) {}

  Expected<ExprPtr> parse_all() {
    auto e = parse_or();
    if (!e) {
      return e;
    }
    if (peek().kind != Token::Kind::End) {
      return fail(fmt::format("unexpected '{}'", peek().text));
    }
    return e;
  }

  // For `set name = expr` and `for name in expr`, the caller consumes the prefix.
  const Token& peek() const { return tokens_[pos_]; }
  Token take() { return tokens_[pos_++]; }
  bool accept_name(std::string_view word) {
    if (peek().kind == Token::Kind::Name && peek().text == word) {
      ++pos_;
      return true;
    }
    return false;
  }
  bool accept_op(std::string_view op) {
    if (peek().kind == Token::Kind::Op && peek().text == op) {
      ++pos_;
      return true;
    }
    return false;
  }
  Error fail(const std::string& what) const {
    return make_error(ErrorCode::Parse, fmt::format("jinja: line {}: {}", line_, what));
  }

  Expected<ExprPtr> parse_or() {
    auto left = parse_and();
    if (!left) {
      return left;
    }
    while (accept_name("or")) {
      auto right = parse_and();
      if (!right) {
        return right;
      }
      left = binary(Expr::Kind::Or, std::move(*left), std::move(*right));
    }
    return left;
  }

private:
  static ExprPtr binary(Expr::Kind kind, ExprPtr l, ExprPtr r) {
    auto e = std::make_unique<Expr>();
    e->kind = kind;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
  }

  Expected<ExprPtr> parse_and() {
    auto left = parse_not();
    if (!left) {
      return left;
    }
    while (accept_name("and")) {
      auto right = parse_not();
      if (!right) {
        return right;
      }
      left = binary(Expr::Kind::And, std::move(*left), std::move(*right));
    }
    return left;
  }

  Expected<ExprPtr> parse_not() {
    if (accept_name("not")) {
      auto inner = parse_not();
      if (!inner) {
        return inner;
      }
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::Not;
      e->left = std::move(*inner);
      return e;
    }
    return parse_compare();
  }

  Expected<ExprPtr> parse_compare() {
    auto left = parse_add();
    if (!left) {
      return left;
    }
    while (true) {
      if (accept_op("==") || accept_op("!=")) {
        const bool eq = tokens_[pos_ - 1].text == "==";
        auto right = parse_add();
        if (!right) {
          return right;
        }
        left = binary(eq ? Expr::Kind::Eq : Expr::Kind::Ne, std::move(*left), std::move(*right));
      } else if (accept_name("in")) {
        auto right = parse_add();
        if (!right) {
          return right;
        }
        left = binary(Expr::Kind::In, std::move(*left), std::move(*right));
      } else if (peek().kind == Token::Kind::Name && peek().text == "not" &&
                 pos_ + 1 < tokens_.size() && tokens_[pos_ + 1].text == "in") {
        pos_ += 2;
        auto right = parse_add();
        if (!right) {
          return right;
        }
        left = binary(Expr::Kind::NotIn, std::move(*left), std::move(*right));
      } else if (accept_name("is")) {
        const bool negate = accept_name("not");
        if (peek().kind != Token::Kind::Name) {
          return fail("expected a test name after 'is'");
        }
        const std::string test = take().text;
        static constexpr std::string_view kTests[] = {"defined", "none", "mapping", "iterable"};
        if (std::ranges::find(kTests, test) == std::end(kTests)) {
          return fail(fmt::format("unsupported test '{}'", test));
        }
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::Test;
        e->name = test;
        e->negate = negate;
        e->left = std::move(*left);
        left = std::move(e);
      } else {
        return left;
      }
    }
  }

  Expected<ExprPtr> parse_add() {
    auto left = parse_postfix();
    if (!left) {
      return left;
    }
    while (accept_op("+") || accept_op("-")) {
      const bool add = tokens_[pos_ - 1].text == "+";
      auto right = parse_postfix();
      if (!right) {
        return right;
      }
      left = binary(add ? Expr::Kind::Add : Expr::Kind::Sub, std::move(*left), std::move(*right));
    }
    return left;
  }

  Expected<std::vector<ExprPtr>> parse_args() {
    std::vector<ExprPtr> args;
    if (accept_op(")")) {
      return args;
    }
    while (true) {
      // Keyword arguments like tojson(indent=4) are accepted syntactically and ignored by
      // value, since the only filters that take them are unsupported anyway.
      if (peek().kind == Token::Kind::Name && pos_ + 1 < tokens_.size() &&
          tokens_[pos_ + 1].kind == Token::Kind::Op && tokens_[pos_ + 1].text == "=") {
        pos_ += 2;
      }
      auto arg = parse_or();
      if (!arg) {
        return arg.error();
      }
      args.push_back(std::move(*arg));
      if (accept_op(")")) {
        return args;
      }
      if (!accept_op(",")) {
        return fail("expected ',' or ')' in argument list");
      }
    }
  }

  Expected<ExprPtr> parse_postfix() {
    auto base = parse_primary();
    if (!base) {
      return base;
    }
    ExprPtr e = std::move(*base);
    while (true) {
      if (accept_op(".")) {
        if (peek().kind != Token::Kind::Name) {
          return fail("expected an attribute name after '.'");
        }
        auto attr = std::make_unique<Expr>();
        attr->kind = Expr::Kind::Attr;
        attr->name = take().text;
        attr->left = std::move(e);
        e = std::move(attr);
      } else if (accept_op("[")) {
        ExprPtr start;
        ExprPtr stop;
        bool slice = false;
        if (!accept_op(":")) {
          auto s = parse_or();
          if (!s) {
            return s;
          }
          start = std::move(*s);
          slice = accept_op(":");
        } else {
          slice = true;
        }
        if (slice && !(peek().kind == Token::Kind::Op && peek().text == "]")) {
          auto s = parse_or();
          if (!s) {
            return s;
          }
          stop = std::move(*s);
        }
        if (!accept_op("]")) {
          return fail("expected ']'");
        }
        auto idx = std::make_unique<Expr>();
        idx->kind = slice ? Expr::Kind::Slice : Expr::Kind::Index;
        idx->left = std::move(e);
        idx->right = std::move(start);
        idx->third = std::move(stop);
        e = std::move(idx);
      } else if (accept_op("|")) {
        if (peek().kind != Token::Kind::Name) {
          return fail("expected a filter name after '|'");
        }
        auto filter = std::make_unique<Expr>();
        filter->kind = Expr::Kind::Filter;
        filter->name = take().text;
        filter->left = std::move(e);
        if (accept_op("(")) {
          auto args = parse_args();
          if (!args) {
            return args.error();
          }
          filter->args = std::move(*args);
        }
        e = std::move(filter);
      } else if (e->kind == Expr::Kind::Name && accept_op("(")) {
        auto call = std::make_unique<Expr>();
        call->kind = Expr::Kind::Call;
        call->name = e->name;
        auto args = parse_args();
        if (!args) {
          return args.error();
        }
        call->args = std::move(*args);
        e = std::move(call);
      } else {
        return e;
      }
    }
  }

  Expected<ExprPtr> parse_primary() {
    const Token t = take();
    auto e = std::make_unique<Expr>();
    switch (t.kind) {
    case Token::Kind::String:
      e->literal = Value::string(t.text);
      return e;
    case Token::Kind::Int:
      e->literal = Value::integer(std::stoll(t.text));
      return e;
    case Token::Kind::Name:
      if (t.text == "true" || t.text == "True") {
        e->literal = Value::boolean(true);
      } else if (t.text == "false" || t.text == "False") {
        e->literal = Value::boolean(false);
      } else if (t.text == "none" || t.text == "None") {
        e->literal = Value::none();
      } else {
        e->kind = Expr::Kind::Name;
        e->name = t.text;
      }
      return e;
    case Token::Kind::Op:
      if (t.text == "-") {
        auto operand = parse_postfix();
        if (!operand) {
          return operand;
        }
        auto zero = std::make_unique<Expr>();
        zero->literal = Value::integer(0);
        return binary(Expr::Kind::Sub, std::move(zero), std::move(*operand));
      }
      if (t.text == "(") {
        auto inner = parse_or();
        if (!inner) {
          return inner;
        }
        if (!accept_op(")")) {
          return fail("expected ')'");
        }
        return inner;
      }
      return fail(fmt::format("unexpected '{}'", t.text));
    case Token::Kind::End:
      return fail("unexpected end of expression");
    }
    return fail("bad token");
  }

  std::vector<Token> tokens_;
  std::size_t pos_ = 0;
  std::size_t line_;
};

Expected<ExprPtr> parse_expression(std::string_view src, std::size_t line) {
  auto tokens = lex_expression(src, line);
  if (!tokens) {
    return tokens.error();
  }
  ExprParser parser(std::move(*tokens), line);
  return parser.parse_all();
}

// --- statement parser ---------------------------------------------------------------------

std::string first_word(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size() && is_space(s[i])) {
    ++i;
  }
  std::size_t j = i;
  while (j < s.size() && !is_space(s[j])) {
    ++j;
  }
  return std::string(s.substr(i, j - i));
}

std::string_view after_word(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size() && is_space(s[i])) {
    ++i;
  }
  while (i < s.size() && !is_space(s[i])) {
    ++i;
  }
  return s.substr(i);
}

class StatementParser {
public:
  explicit StatementParser(std::vector<Chunk> chunks) : chunks_(std::move(chunks)) {}

  Expected<Body> parse_body(std::vector<std::string_view> terminators, std::string* ended_by) {
    Body body;
    while (pos_ < chunks_.size()) {
      const Chunk& chunk = chunks_[pos_];
      if (chunk.kind == Chunk::Kind::Comment) {
        ++pos_;
        continue;
      }
      if (chunk.kind == Chunk::Kind::Text) {
        auto node = std::make_unique<Node>();
        node->kind = Node::Kind::Text;
        node->text = chunk.content;
        body.push_back(std::move(node));
        ++pos_;
        continue;
      }
      if (chunk.kind == Chunk::Kind::Output) {
        auto expr = parse_expression(chunk.content, chunk.line);
        if (!expr) {
          return expr.error();
        }
        auto node = std::make_unique<Node>();
        node->kind = Node::Kind::Output;
        node->expr = std::move(*expr);
        body.push_back(std::move(node));
        ++pos_;
        continue;
      }
      const std::string word = first_word(chunk.content);
      if (std::ranges::find(terminators, word) != terminators.end()) {
        if (ended_by != nullptr) {
          *ended_by = word;
        }
        return body;  // the caller consumes the terminator
      }
      if (word == "if") {
        auto node = parse_if();
        if (!node) {
          return node.error();
        }
        body.push_back(std::move(*node));
      } else if (word == "for") {
        auto node = parse_for();
        if (!node) {
          return node.error();
        }
        body.push_back(std::move(*node));
      } else if (word == "set") {
        auto node = parse_set();
        if (!node) {
          return node.error();
        }
        body.push_back(std::move(*node));
      } else {
        return make_error(ErrorCode::Parse,
                          fmt::format("jinja: line {}: unsupported tag '{}'", chunk.line, word));
      }
    }
    if (!terminators.empty()) {
      return make_error(ErrorCode::Parse,
                        fmt::format("jinja: missing '{}' at end of template", terminators[0]));
    }
    return body;
  }

private:
  Expected<std::unique_ptr<Node>> parse_if() {
    auto node = std::make_unique<Node>();
    node->kind = Node::Kind::If;
    while (true) {
      const Chunk& head = chunks_[pos_++];
      auto cond = parse_expression(after_word(head.content), head.line);
      if (!cond) {
        return cond.error();
      }
      std::string ended;
      auto body = parse_body({"elif", "else", "endif"}, &ended);
      if (!body) {
        return body.error();
      }
      node->branches.emplace_back(std::move(*cond), std::move(*body));
      if (ended == "elif") {
        continue;
      }
      if (ended == "else") {
        ++pos_;
        auto tail = parse_body({"endif"}, nullptr);
        if (!tail) {
          return tail.error();
        }
        node->else_body = std::move(*tail);
      }
      ++pos_;  // endif
      return node;
    }
  }

  Expected<std::unique_ptr<Node>> parse_for() {
    const Chunk& head = chunks_[pos_++];
    const std::string_view rest = after_word(head.content);
    const std::string var = first_word(rest);
    const std::string_view after_var = after_word(rest);
    if (first_word(after_var) != "in") {
      return make_error(ErrorCode::Parse,
                        fmt::format("jinja: line {}: expected 'for <name> in <expr>'", head.line));
    }
    auto iterable = parse_expression(after_word(after_var), head.line);
    if (!iterable) {
      return iterable.error();
    }
    auto node = std::make_unique<Node>();
    node->kind = Node::Kind::For;
    node->name = var;
    node->expr = std::move(*iterable);
    auto body = parse_body({"endfor"}, nullptr);
    if (!body) {
      return body.error();
    }
    node->body = std::move(*body);
    ++pos_;  // endfor
    return node;
  }

  Expected<std::unique_ptr<Node>> parse_set() {
    const Chunk& head = chunks_[pos_++];
    const std::string_view rest = after_word(head.content);
    const auto eq = rest.find('=');
    if (eq == std::string_view::npos) {
      return make_error(ErrorCode::Parse,
                        fmt::format("jinja: line {}: expected 'set <name> = <expr>'", head.line));
    }
    std::string name = first_word(rest.substr(0, eq));
    auto expr = parse_expression(rest.substr(eq + 1), head.line);
    if (!expr) {
      return expr.error();
    }
    auto node = std::make_unique<Node>();
    node->kind = Node::Kind::Set;
    node->name = std::move(name);
    node->expr = std::move(*expr);
    return node;
  }

  std::vector<Chunk> chunks_;
  std::size_t pos_ = 0;
};

// --- evaluation ---------------------------------------------------------------------------

class Renderer {
public:
  explicit Renderer(const Map& variables) { scopes_.push_back(variables); }

  Expected<bool> render(const Body& body) {
    for (const auto& node : body) {
      auto ok = render_node(*node);
      if (!ok) {
        return ok;
      }
    }
    return true;
  }
  std::string take_output() { return std::move(out_); }

private:
  static Error fail(const std::string& what) {
    return make_error(ErrorCode::InvalidArgument, fmt::format("jinja: {}", what));
  }

  const Value* lookup(std::string_view name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      for (const auto& [k, v] : *it) {
        if (k == name) {
          return &v;
        }
      }
    }
    return nullptr;
  }

  void assign(const std::string& name, Value value) {
    Map& scope = scopes_.back();
    for (auto& [k, v] : scope) {
      if (k == name) {
        v = std::move(value);
        return;
      }
    }
    scope.emplace_back(name, std::move(value));
  }

  Expected<bool> render_node(const Node& node) {
    switch (node.kind) {
    case Node::Kind::Text:
      out_ += node.text;
      return true;
    case Node::Kind::Output: {
      auto value = eval(*node.expr);
      if (!value) {
        return value.error();
      }
      return emit(*value);
    }
    case Node::Kind::If: {
      for (const auto& [cond, body] : node.branches) {
        auto value = eval(*cond);
        if (!value) {
          return value.error();
        }
        if (value->truthy()) {
          return render(body);
        }
      }
      return render(node.else_body);
    }
    case Node::Kind::For: {
      auto iterable = eval(*node.expr);
      if (!iterable) {
        return iterable.error();
      }
      List items;
      if (iterable->kind() == Value::Kind::List) {
        items = iterable->as_list();
      } else if (iterable->kind() == Value::Kind::Undefined ||
                 iterable->kind() == Value::Kind::None) {
        items = {};
      } else {
        return fail(fmt::format("cannot iterate over a {}", iterable->kind_name()));
      }
      const auto n = static_cast<std::int64_t>(items.size());
      for (std::int64_t i = 0; i < n; ++i) {
        Map scope;
        scope.emplace_back(node.name, items[static_cast<std::size_t>(i)]);
        scope.emplace_back("loop", Value::map({{"first", Value::boolean(i == 0)},
                                               {"last", Value::boolean(i + 1 == n)},
                                               {"index0", Value::integer(i)},
                                               {"index", Value::integer(i + 1)},
                                               {"length", Value::integer(n)}}));
        scopes_.push_back(std::move(scope));
        auto ok = render(node.body);
        scopes_.pop_back();
        if (!ok) {
          return ok;
        }
      }
      return true;
    }
    case Node::Kind::Set: {
      auto value = eval(*node.expr);
      if (!value) {
        return value.error();
      }
      assign(node.name, std::move(*value));
      return true;
    }
    }
    return true;
  }

  Expected<bool> emit(const Value& v) {
    switch (v.kind()) {
    case Value::Kind::Undefined:
    case Value::Kind::None:
      return true;
    case Value::Kind::Bool:
      out_ += v.as_bool() ? "True" : "False";
      return true;
    case Value::Kind::Int:
      out_ += std::to_string(v.as_int());
      return true;
    case Value::Kind::String:
      out_ += v.as_string();
      return true;
    case Value::Kind::List:
    case Value::Kind::Map:
      return fail(fmt::format("cannot print a {}", v.kind_name()));
    }
    return true;
  }

  static bool is_builtin(std::string_view name) {
    return name == "strftime_now" || name == "raise_exception";
  }

  Expected<Value> eval(const Expr& e) {
    switch (e.kind) {
    case Expr::Kind::Literal:
      return e.literal;
    case Expr::Kind::Name: {
      const Value* v = lookup(e.name);
      return v != nullptr ? *v : Value{};
    }
    case Expr::Kind::Attr: {
      auto obj = eval(*e.left);
      if (!obj) {
        return obj;
      }
      const Value* v = obj->get(e.name);
      return v != nullptr ? *v : Value{};
    }
    case Expr::Kind::Index: {
      auto obj = eval(*e.left);
      if (!obj) {
        return obj;
      }
      auto key = eval(*e.right);
      if (!key) {
        return key;
      }
      if (key->kind() == Value::Kind::String) {
        const Value* v = obj->get(key->as_string());
        return v != nullptr ? *v : Value{};
      }
      if (key->kind() == Value::Kind::Int && obj->kind() == Value::Kind::List) {
        const auto& items = obj->as_list();
        std::int64_t i = key->as_int();
        if (i < 0) {
          i += static_cast<std::int64_t>(items.size());
        }
        if (i < 0 || i >= static_cast<std::int64_t>(items.size())) {
          return Value{};
        }
        return items[static_cast<std::size_t>(i)];
      }
      return fail(fmt::format("cannot index a {} with a {}", obj->kind_name(), key->kind_name()));
    }
    case Expr::Kind::Slice: {
      auto obj = eval(*e.left);
      if (!obj) {
        return obj;
      }
      if (obj->kind() != Value::Kind::List) {
        return fail(fmt::format("cannot slice a {}", obj->kind_name()));
      }
      const auto& items = obj->as_list();
      const auto n = static_cast<std::int64_t>(items.size());
      std::int64_t start = 0;
      std::int64_t stop = n;
      if (e.right) {
        auto s = eval(*e.right);
        if (!s) {
          return s;
        }
        start = s->as_int();
      }
      if (e.third) {
        auto s = eval(*e.third);
        if (!s) {
          return s;
        }
        stop = s->as_int();
      }
      if (start < 0) {
        start += n;
      }
      if (stop < 0) {
        stop += n;
      }
      start = std::clamp<std::int64_t>(start, 0, n);
      stop = std::clamp<std::int64_t>(stop, start, n);
      return Value::list(List(items.begin() + start, items.begin() + stop));
    }
    case Expr::Kind::Add:
    case Expr::Kind::Sub: {
      auto l = eval(*e.left);
      if (!l) {
        return l;
      }
      auto r = eval(*e.right);
      if (!r) {
        return r;
      }
      const bool add = e.kind == Expr::Kind::Add;
      if (add && l->kind() == Value::Kind::String && r->kind() == Value::Kind::String) {
        return Value::string(l->as_string() + r->as_string());
      }
      if (l->kind() == Value::Kind::Int && r->kind() == Value::Kind::Int) {
        return Value::integer(add ? l->as_int() + r->as_int() : l->as_int() - r->as_int());
      }
      return fail(fmt::format("cannot {} a {} and a {}", add ? "add" : "subtract", l->kind_name(),
                              r->kind_name()));
    }
    case Expr::Kind::Eq:
    case Expr::Kind::Ne: {
      auto l = eval(*e.left);
      if (!l) {
        return l;
      }
      auto r = eval(*e.right);
      if (!r) {
        return r;
      }
      const bool equal = l->equals(*r);
      return Value::boolean(e.kind == Expr::Kind::Eq ? equal : !equal);
    }
    case Expr::Kind::And: {
      auto l = eval(*e.left);
      if (!l) {
        return l;
      }
      if (!l->truthy()) {
        return Value::boolean(false);
      }
      auto r = eval(*e.right);
      if (!r) {
        return r;
      }
      return Value::boolean(r->truthy());
    }
    case Expr::Kind::Or: {
      auto l = eval(*e.left);
      if (!l) {
        return l;
      }
      if (l->truthy()) {
        return Value::boolean(true);
      }
      auto r = eval(*e.right);
      if (!r) {
        return r;
      }
      return Value::boolean(r->truthy());
    }
    case Expr::Kind::Not: {
      auto v = eval(*e.left);
      if (!v) {
        return v;
      }
      return Value::boolean(!v->truthy());
    }
    case Expr::Kind::In:
    case Expr::Kind::NotIn: {
      auto item = eval(*e.left);
      if (!item) {
        return item;
      }
      auto container = eval(*e.right);
      if (!container) {
        return container;
      }
      bool found = false;
      if (container->kind() == Value::Kind::Map && item->kind() == Value::Kind::String) {
        found = container->get(item->as_string()) != nullptr;
      } else if (container->kind() == Value::Kind::List) {
        found = std::ranges::any_of(container->as_list(),
                                    [&](const Value& v) { return v.equals(*item); });
      } else if (container->kind() == Value::Kind::String && item->kind() == Value::Kind::String) {
        found = container->as_string().find(item->as_string()) != std::string::npos;
      } else if (container->kind() == Value::Kind::Undefined ||
                 container->kind() == Value::Kind::None) {
        found = false;
      } else {
        return fail(fmt::format("cannot test membership in a {}", container->kind_name()));
      }
      return Value::boolean(e.kind == Expr::Kind::In ? found : !found);
    }
    case Expr::Kind::Test: {
      bool result = false;
      if (e.name == "defined") {
        if (e.left->kind == Expr::Kind::Name) {
          result = lookup(e.left->name) != nullptr || is_builtin(e.left->name);
        } else {
          auto v = eval(*e.left);
          if (!v) {
            return v;
          }
          result = v->kind() != Value::Kind::Undefined;
        }
      } else {
        auto v = eval(*e.left);
        if (!v) {
          return v;
        }
        if (e.name == "none") {
          result = v->kind() == Value::Kind::None;
        } else if (e.name == "mapping") {
          result = v->kind() == Value::Kind::Map;
        } else {  // iterable
          result = v->kind() == Value::Kind::List || v->kind() == Value::Kind::Map ||
                   v->kind() == Value::Kind::String;
        }
      }
      return Value::boolean(e.negate ? !result : result);
    }
    case Expr::Kind::Filter: {
      auto v = eval(*e.left);
      if (!v) {
        return v;
      }
      if (e.name == "trim") {
        if (v->kind() != Value::Kind::String) {
          return fail(fmt::format("trim expects a string, found a {}", v->kind_name()));
        }
        std::string_view s = v->as_string();
        while (!s.empty() && is_space(s.front())) {
          s.remove_prefix(1);
        }
        while (!s.empty() && is_space(s.back())) {
          s.remove_suffix(1);
        }
        return Value::string(std::string(s));
      }
      if (e.name == "length") {
        switch (v->kind()) {
        case Value::Kind::String:
          return Value::integer(static_cast<std::int64_t>(v->as_string().size()));
        case Value::Kind::List:
          return Value::integer(static_cast<std::int64_t>(v->as_list().size()));
        case Value::Kind::Map:
          return Value::integer(static_cast<std::int64_t>(v->as_map().size()));
        default:
          return fail(fmt::format("length of a {}", v->kind_name()));
        }
      }
      return fail(fmt::format("unsupported filter '{}'", e.name));
    }
    case Expr::Kind::Call: {
      if (e.name == "raise_exception") {
        std::string message = "template raised an exception";
        if (!e.args.empty()) {
          auto v = eval(*e.args[0]);
          if (!v) {
            return v;
          }
          if (v->kind() == Value::Kind::String) {
            message = v->as_string();
          }
        }
        return fail(message);
      }
      if (e.name == "strftime_now") {
        std::string format = "%Y-%m-%d";
        if (!e.args.empty()) {
          auto v = eval(*e.args[0]);
          if (!v) {
            return v;
          }
          if (v->kind() == Value::Kind::String) {
            format = v->as_string();
          }
        }
        const std::time_t now = std::time(nullptr);
        std::tm local{};
        localtime_r(&now, &local);
        char buffer[128];
        const std::size_t n = std::strftime(buffer, sizeof buffer, format.c_str(), &local);
        return Value::string(std::string(buffer, n));
      }
      return fail(fmt::format("unsupported function '{}'", e.name));
    }
    }
    return fail("bad expression");
  }

  std::vector<Map> scopes_;
  std::string out_;
};

}  // namespace

// --- Template -----------------------------------------------------------------------------

Template::Template(std::vector<std::unique_ptr<Node>> body) : body_(std::move(body)) {}
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;
Template::~Template() = default;

Expected<Template> Template::parse(std::string_view source) {
  auto chunks = lex_template(source);
  if (!chunks) {
    return chunks.error();
  }
  StatementParser parser(std::move(*chunks));
  auto body = parser.parse_body({}, nullptr);
  if (!body) {
    return body.error();
  }
  return Template(std::move(*body));
}

Expected<std::string> Template::render(const Map& variables) const {
  Renderer renderer(variables);
  auto ok = renderer.render(body_);
  if (!ok) {
    return ok.error();
  }
  return renderer.take_output();
}

}  // namespace axon::chat
