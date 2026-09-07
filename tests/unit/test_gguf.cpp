#include "core/dtype.h"
#include "core/gguf.h"

#include "fixtures.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

axon::GGUFFile open_fixture(const std::string& name) {
  auto file = axon::GGUFFile::open(axon::test::fixture_path("gguf/" + name));
  if (!file) {
    FAIL("cannot open fixture " << name << ": " << file.error().message);
  }
  return std::move(*file);
}

}  // namespace

TEST_CASE("tiny fixture header", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");
  CHECK(file.version() == 3);
  CHECK(file.alignment() == 64);
  CHECK(file.data_offset() % 64 == 0);
  CHECK(file.data_offset() < file.file_size());
  CHECK(file.tensor_count() == 5);
  CHECK(file.architecture() == "axon-test");
  CHECK(file.model_name() == "tiny fixture");
  CHECK(file.metadata_entries().size() == 22);
  CHECK(file.tensor_data_bytes() == 128 + 128 + 64 + 204 + 72);
}

TEST_CASE("tiny fixture scalar metadata of every type", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");
  CHECK(file.get_uint("test.u8").value() == 200);
  CHECK(file.metadata("test.u8")->type() == axon::GGUFValueType::UInt8);
  CHECK(file.get_int("test.i8").value() == -7);
  CHECK(file.get_uint("test.u16").value() == 65535);
  CHECK(file.get_int("test.i16").value() == -32768);
  CHECK(file.get_uint("test.u32").value() == 4000000000ULL);
  CHECK(file.get_int("test.i32").value() == -2000000000);
  CHECK(file.get_float("test.f32").value() == 1.5);
  CHECK(file.get_bool("test.bool_true").value());
  CHECK_FALSE(file.get_bool("test.bool_false").value());
  CHECK(file.get_string("test.string").value() == "héllo wörld ✓");
  CHECK(file.get_string("test.empty_string").value().empty());
  CHECK(file.get_uint("test.u64").value() == 18446744073709551615ULL);
  CHECK(file.get_int("test.i64").value() == -9223372036854775807LL - 1);
  CHECK(file.get_float("test.f64").value() == -2.5e-300);

  // Unsigned values are readable as signed when they fit; the reverse is never allowed.
  CHECK(file.get_int("test.u32").value() == 4000000000LL);
  CHECK_FALSE(file.get_int("test.u64"));
  CHECK_FALSE(file.get_uint("test.i8"));
  CHECK(file.get_uint_or("test.i8", 9) == 9);
  CHECK(file.get_float_or("nope", 2.0) == 2.0);
  CHECK(file.get_string_or("nope", "fallback") == "fallback");
}

TEST_CASE("tiny fixture arrays", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");

  const auto* strings = file.metadata("test.strings");
  REQUIRE(strings != nullptr);
  REQUIRE(strings->is_array());
  CHECK(strings->array_element_type() == axon::GGUFValueType::String);
  auto sa = strings->as_string_array();
  REQUIRE(sa);
  REQUIRE(sa->size() == 3);
  CHECK((*sa)[0] == "alpha");
  CHECK((*sa)[1].empty());
  CHECK((*sa)[2] == "gamma δ");
  CHECK(strings->as_array() == nullptr);
  CHECK_FALSE(strings->as_float_array());

  auto fa = file.metadata("test.f32s")->as_float_array();
  REQUIRE(fa);
  REQUIRE(fa->size() == 3);
  CHECK((*fa)[0] == 0.25F);
  CHECK((*fa)[1] == -1.0F);
  CHECK((*fa)[2] == 3.0e10F);

  auto ia = file.metadata("test.i32s")->as_int32_array();
  REQUIRE(ia);
  CHECK((*ia)[0] == -1);
  CHECK((*ia)[2] == 2147483647);

  const auto* u8s = file.metadata("test.u8s")->as_array();
  REQUIRE(u8s != nullptr);
  REQUIRE(u8s->size() == 3);
  CHECK((*u8s)[2].as_uint().value() == 255);

  const auto* nested = file.metadata("test.nested");
  REQUIRE(nested != nullptr);
  CHECK(nested->array_element_type() == axon::GGUFValueType::Array);
  const auto* outer = nested->as_array();
  REQUIRE(outer != nullptr);
  REQUIRE(outer->size() == 2);
  REQUIRE((*outer)[1].is_array());
  CHECK((*outer)[0].array_size() == 2);
  CHECK((*outer)[1].array_size() == 3);
  const auto inner = (*outer)[1].as_int32_array();  // int elements are stored typed
  REQUIRE(inner);
  CHECK((*inner)[2] == 5);
}

namespace {

void put_u32(std::vector<std::byte>& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
  }
}

void put_u64(std::vector<std::byte>& out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
  }
}

void put_string(std::vector<std::byte>& out, std::string_view s) {
  put_u64(out, s.size());
  for (const char c : s) {
    out.push_back(static_cast<std::byte>(c));
  }
}

// A minimal file: header, the given raw metadata entries, no tensors.
std::vector<std::byte> minimal_gguf(const std::vector<std::vector<std::byte>>& entries) {
  std::vector<std::byte> out;
  put_u32(out, 0x46554747U);
  put_u32(out, 3);
  put_u64(out, 0);
  put_u64(out, entries.size());
  for (const auto& e : entries) {
    out.insert(out.end(), e.begin(), e.end());
  }
  while (out.size() % 32 != 0) {
    out.push_back(std::byte{0});
  }
  return out;
}

}  // namespace

TEST_CASE("hand-built files: empty arrays, duplicate keys, bad alignment", "[core][gguf]") {
  std::vector<std::byte> empty_array;
  put_string(empty_array, "test.empty");
  put_u32(empty_array, 9);  // array
  put_u32(empty_array, 4);  // of u32
  put_u64(empty_array, 0);  // none

  SECTION("empty array parses with size zero") {
    const auto bytes = minimal_gguf({empty_array});
    const auto file = axon::GGUFFile::parse(bytes);
    REQUIRE(file);
    const auto* v = file->metadata("test.empty");
    REQUIRE(v != nullptr);
    CHECK(v->is_array());
    CHECK(v->array_size() == 0);
    CHECK(v->array_element_type() == axon::GGUFValueType::UInt32);
    CHECK(file->tensor_count() == 0);
    CHECK(file->data_offset() == bytes.size());
  }
  SECTION("duplicate keys are rejected") {
    const auto bytes = minimal_gguf({empty_array, empty_array});
    const auto file = axon::GGUFFile::parse(bytes);
    REQUIRE_FALSE(file);
    CHECK_THAT(file.error().message, ContainsSubstring("duplicate key 'test.empty'"));
  }
  SECTION("alignment must be a u32 power of two") {
    std::vector<std::byte> align;
    put_string(align, "general.alignment");
    put_u32(align, 4);   // u32
    put_u32(align, 48);  // not a power of two
    const auto file = axon::GGUFFile::parse(minimal_gguf({align}));
    REQUIRE_FALSE(file);
    CHECK(file.error().message == "general.alignment: expected a power of two, found 48");
  }
  SECTION("unknown value type is rejected") {
    std::vector<std::byte> bad;
    put_string(bad, "x");
    put_u32(bad, 13);
    const auto file = axon::GGUFFile::parse(minimal_gguf({bad}));
    REQUIRE_FALSE(file);
    CHECK_THAT(file.error().message, ContainsSubstring("unknown value type 13"));
  }
}

TEST_CASE("missing and mistyped metadata produce expected/found messages", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");
  CHECK(file.metadata("does.not.exist") == nullptr);
  const auto missing = file.get_uint("does.not.exist");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().message == "metadata 'does.not.exist': expected present, found missing");
  const auto wrong = file.get_string("test.u8");
  REQUIRE_FALSE(wrong);
  CHECK(wrong.error().message == "metadata 'test.u8': expected a string, found u8");
}

TEST_CASE("tiny fixture tensor table", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");
  const auto tensors = file.tensors();
  REQUIRE(tensors.size() == 5);
  CHECK(tensors[0].name == "t.f32");
  CHECK(tensors[1].name == "t.f16");
  CHECK(tensors[2].name == "t.bf16");
  CHECK(tensors[3].name == "t.q8_0");
  CHECK(tensors[4].name == "t.q4_0");
  CHECK(file.tensor("nope") == nullptr);

  const auto* f32 = file.tensor("t.f32");
  REQUIRE(f32 != nullptr);
  CHECK(f32->dtype == axon::DType::F32);
  CHECK(f32->n_dims == 2);
  CHECK(f32->shape[0] == 8);  // innermost: columns
  CHECK(f32->shape[1] == 4);
  CHECK(f32->ne(2) == 1);
  CHECK(f32->n_elements() == 32);
  CHECK(f32->nbytes == 128);
  CHECK(f32->offset == 0);

  const auto* q8 = file.tensor("t.q8_0");
  REQUIRE(q8 != nullptr);
  CHECK(q8->dtype == axon::DType::Q8_0);
  CHECK(q8->shape[0] == 64);
  CHECK(q8->shape[1] == 3);
  CHECK(q8->nbytes == 204);
  CHECK(q8->offset % 64 == 0);

  for (const auto& t : tensors) {
    CHECK(reinterpret_cast<std::uintptr_t>(t.data) % 64 == 0);
    CHECK(t.offset + t.nbytes <= file.file_size() - file.data_offset());
  }
}

TEST_CASE("tiny fixture tensor contents", "[core][gguf]") {
  const auto file = open_fixture("tiny.gguf");

  const auto* f32 = file.tensor("t.f32");
  const auto* values = reinterpret_cast<const float*>(f32->data);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 8; ++c) {
      CHECK(values[r * 8 + c] == static_cast<float>(r * 8 + c));
    }
  }

  const auto* f16 = file.tensor("t.f16");
  const auto* halves = reinterpret_cast<const std::uint16_t*>(f16->data);
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 32; ++c) {
      CHECK(axon::fp16_to_fp32(halves[r * 32 + c]) ==
            static_cast<float>(c) - 16.0F + static_cast<float>(r) * 0.5F);
    }
  }

  const auto* bf16 = file.tensor("t.bf16");
  const auto* bhalves = reinterpret_cast<const std::uint16_t*>(bf16->data);
  for (int i = 0; i < 32; ++i) {
    CHECK(axon::bf16_to_fp32(bhalves[i]) == std::ldexp(1.0F, i - 16));
  }

  const auto* q8 = file.tensor("t.q8_0");
  for (int b = 0; b < 6; ++b) {
    axon::BlockQ8_0 block{};
    std::memcpy(&block, q8->data + static_cast<std::size_t>(b) * sizeof(axon::BlockQ8_0),
                sizeof(block));
    CHECK(axon::fp16_to_fp32(block.d) == static_cast<float>(b + 1) / 64.0F);
    for (int i = 0; i < 32; ++i) {
      CHECK(block.qs[i] == static_cast<std::int8_t>(i - b * 4 - 16));
    }
  }

  const auto* q4 = file.tensor("t.q4_0");
  for (int b = 0; b < 4; ++b) {
    axon::BlockQ4_0 block{};
    std::memcpy(&block, q4->data + static_cast<std::size_t>(b) * sizeof(axon::BlockQ4_0),
                sizeof(block));
    CHECK(axon::fp16_to_fp32(block.d) == 0.5F * static_cast<float>(b + 1));
    for (int i = 0; i < 16; ++i) {
      CHECK((block.qs[i] & 0x0F) == (i + b) % 16);     // element i
      CHECK((block.qs[i] >> 4) == (15 - i + b) % 16);  // element i + 16
    }
  }
}

TEST_CASE("synthetic model fixture", "[core][gguf]") {
  const auto file = open_fixture("synthetic-2l.gguf");
  CHECK(file.architecture() == "llama");
  CHECK(file.get_uint("llama.block_count").value() == 2);
  CHECK(file.get_uint("llama.embedding_length").value() == 64);
  CHECK(file.get_uint("llama.attention.head_count").value() == 4);
  CHECK(file.get_uint("llama.attention.head_count_kv").value() == 2);
  CHECK(file.get_uint("llama.feed_forward_length").value() == 160);
  CHECK(file.get_uint("llama.context_length").value() == 128);
  CHECK(file.get_float("llama.rope.freq_base").value() == 10000.0);
  CHECK(file.get_string("tokenizer.ggml.model").value() == "gpt2");
  CHECK(file.metadata("tokenizer.ggml.tokens")->array_size() == 512);
  CHECK(file.metadata("tokenizer.ggml.token_type")->as_int32_array()->size() == 512);
  CHECK(file.metadata("tokenizer.ggml.merges") == nullptr);  // no merges: key omitted
  CHECK(file.get_uint("tokenizer.ggml.bos_token_id").value() == 256);

  CHECK(file.tensor_count() == 20);
  const auto* embd = file.tensor("token_embd.weight");
  REQUIRE(embd != nullptr);
  CHECK(embd->shape[0] == 64);
  CHECK(embd->shape[1] == 512);
  CHECK(embd->nbytes == std::size_t{512} * 64 * 4);
  const auto* down = file.tensor("blk.1.ffn_down.weight");
  REQUIRE(down != nullptr);
  CHECK(down->shape[0] == 160);                    // n_in
  CHECK(down->shape[1] == 64);                     // n_out
  CHECK(file.tensor("output.weight") == nullptr);  // tied embeddings

  // Every tensor lies inside the data section and the last one ends at the file end.
  std::size_t last_end = 0;
  for (const auto& t : file.tensors()) {
    CHECK(t.offset + t.nbytes <= file.file_size() - file.data_offset());
    last_end = std::max(last_end, static_cast<std::size_t>(t.offset + t.nbytes));
  }
  CHECK(file.data_offset() + last_end == file.file_size());
}

TEST_CASE("quantized synthetic fixtures", "[core][gguf]") {
  const auto q8 = open_fixture("synthetic-2l-q8_0.gguf");
  CHECK(q8.tensor("token_embd.weight")->dtype == axon::DType::Q8_0);
  CHECK(q8.tensor("token_embd.weight")->nbytes == std::size_t{512} * 2 * 34);
  CHECK(q8.tensor("output_norm.weight")->dtype == axon::DType::F32);
  const auto q4 = open_fixture("synthetic-2l-q4_0.gguf");
  CHECK(q4.tensor("blk.0.ffn_up.weight")->dtype == axon::DType::Q4_0);
  CHECK(q4.tensor("blk.0.ffn_up.weight")->nbytes == std::size_t{160} * 2 * 18);
}

TEST_CASE("open errors name the file and the problem", "[core][gguf]") {
  const auto missing = axon::GGUFFile::open("/definitely/not/here.gguf");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == axon::ErrorCode::Io);
  CHECK_THAT(missing.error().message, ContainsSubstring("cannot open"));
}

TEST_CASE("parse rejects malformed headers with precise messages", "[core][gguf]") {
  const auto original = axon::test::read_fixture("gguf/tiny.gguf");
  auto parse = [](std::vector<std::byte> bytes) { return axon::GGUFFile::parse(bytes); };

  SECTION("bad magic") {
    auto bytes = original;
    bytes[0] = std::byte{'X'};
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("expected \"GGUF\""));
  }
  SECTION("big-endian magic") {
    auto bytes = original;
    bytes[0] = std::byte{'F'};
    bytes[1] = std::byte{'U'};
    bytes[2] = std::byte{'G'};
    bytes[3] = std::byte{'G'};
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("big-endian"));
  }
  SECTION("version 2") {
    auto bytes = original;
    bytes[4] = std::byte{2};
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK(r.error().message == "gguf version: expected 3, found 2");
  }
  SECTION("truncated header") {
    std::vector<std::byte> bytes(original.begin(), original.begin() + 10);
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("gguf tensor count"));
  }
  SECTION("absurd tensor count") {
    auto bytes = original;
    bytes[8] = std::byte{0xFF};
    bytes[15] = std::byte{0x7F};
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("cannot fit"));
  }
  SECTION("truncated inside the tensor data") {
    std::vector<std::byte> bytes(original.begin(), original.end() - 80);  // past the tail padding
    const auto r = parse(bytes);
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("exceed"));
  }
  SECTION("empty input") {
    const auto r = parse({});
    REQUIRE_FALSE(r);
    CHECK_THAT(r.error().message, ContainsSubstring("gguf magic"));
  }
}
