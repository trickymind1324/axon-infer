// The batched decode step against decoding each sequence alone: the same logits bit for bit,
// on every synthetic format, both kernel tables, several thread counts and both cache types,
// with sequences at unequal positions. Then the bounds the step enforces.

#include "alloc_counter.h"
#include "fixture_tensor.h"
#include "kernels/kernels.h"
#include "model/kv_cache.h"
#include "model/llama.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

std::unique_ptr<axon::model::LlamaModel> load(const std::string& file, int threads,
                                              axon::kernels::KernelSet set,
                                              axon::model::CacheType cache, std::size_t max_batch) {
  auto model = axon::model::LlamaModel::load(axon::test::fixture_path("gguf/" + file),
                                             {.max_seq = 8,
                                              .max_ctx = 32,
                                              .n_threads = threads,
                                              .kernels = set,
                                              .max_batch = max_batch,
                                              .cache = cache});
  if (!model) {
    if (model.error().code == axon::ErrorCode::Unsupported) {
      SKIP(model.error().message);
    }
    FAIL(model.error().message);
  }
  return std::move(*model);
}

// Three sequences with prompts of different lengths, then the same continuation tokens fed
// one step at a time. Returns, per sequence, the logits of every decode step concatenated.
struct Scenario {
  std::vector<std::vector<std::int32_t>> prompts;
  std::vector<std::int32_t> steps;
};

Scenario scenario(const std::vector<std::int32_t>& tokens) {
  return {.prompts = {{tokens[0], tokens[1], tokens[2]},
                      {tokens[3], tokens[4], tokens[5], tokens[6], tokens[7], tokens[8], tokens[9]},
                      {tokens[10]}},
          .steps = {tokens[11], tokens[12], tokens[13], tokens[14]}};
}

std::vector<std::vector<float>> alone(axon::model::LlamaModel& model, const Scenario& sc) {
  std::vector<std::vector<float>> out;
  for (const auto& prompt : sc.prompts) {
    auto cache = model.new_cache();
    REQUIRE(model.forward_prefill(cache, prompt));
    std::vector<float> logits;
    for (const std::int32_t t : sc.steps) {
      const auto step = model.forward_one(cache, t);
      REQUIRE(step);
      logits.insert(logits.end(), step->begin(), step->end());
    }
    out.push_back(std::move(logits));
  }
  return out;
}

std::vector<std::vector<float>> together(axon::model::LlamaModel& model, const Scenario& sc) {
  std::vector<axon::model::KVCache> caches;
  caches.reserve(sc.prompts.size());
  for (const auto& prompt : sc.prompts) {
    caches.push_back(model.new_cache());
    REQUIRE(model.forward_prefill(caches.back(), prompt));
  }
  std::vector<axon::model::KVCache*> ptrs;
  ptrs.reserve(caches.size());
  for (auto& c : caches) {
    ptrs.push_back(&c);
  }
  const std::size_t n_vocab = model.config().n_vocab;
  std::vector<std::vector<float>> out(sc.prompts.size());
  for (const std::int32_t t : sc.steps) {
    const std::vector<std::int32_t> row(sc.prompts.size(), t);
    const auto logits = model.forward_batch(ptrs, row);
    REQUIRE(logits);
    REQUIRE(logits->size() == row.size() * n_vocab);
    for (std::size_t i = 0; i < row.size(); ++i) {
      const auto part = logits->subspan(i * n_vocab, n_vocab);
      out[i].insert(out[i].end(), part.begin(), part.end());
    }
  }
  for (std::size_t i = 0; i < caches.size(); ++i) {
    CHECK(caches[i].pos() == sc.prompts[i].size() + sc.steps.size());
  }
  return out;
}

}  // namespace

TEST_CASE("a batched decode step equals decoding each sequence alone, bit for bit",
          "[model][batch]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  const auto sc = scenario(tokens);
  struct Case {
    const char* file;
    axon::kernels::KernelSet set;
    int threads;
    axon::model::CacheType cache;
  };
  using axon::kernels::KernelSet;
  using axon::model::CacheType;
  for (const auto& c : {Case{.file = "synthetic-2l.gguf",
                             .set = KernelSet::Scalar,
                             .threads = 1,
                             .cache = CacheType::F32},
                        Case{.file = "synthetic-2l.gguf",
                             .set = KernelSet::Auto,
                             .threads = 3,
                             .cache = CacheType::F32},
                        Case{.file = "synthetic-2l.gguf",
                             .set = KernelSet::Auto,
                             .threads = 2,
                             .cache = CacheType::F16},
                        Case{.file = "synthetic-2l-q8_0.gguf",
                             .set = KernelSet::Scalar,
                             .threads = 2,
                             .cache = CacheType::F32},
                        Case{.file = "synthetic-2l-q8_0.gguf",
                             .set = KernelSet::Auto,
                             .threads = 4,
                             .cache = CacheType::F32},
                        Case{.file = "synthetic-2l-q4_0.gguf",
                             .set = KernelSet::Auto,
                             .threads = 3,
                             .cache = CacheType::F32},
                        Case{.file = "synthetic-2l-q4_0.gguf",
                             .set = KernelSet::Auto,
                             .threads = 1,
                             .cache = CacheType::F16},
                        Case{.file = "synthetic-2l-q4_1.gguf",
                             .set = KernelSet::Auto,
                             .threads = 2,
                             .cache = CacheType::F32}}) {
    auto model = load(c.file, c.threads, c.set, c.cache, 3);
    INFO(c.file << " " << axon::kernels::name(model->table()) << " threads " << c.threads
                << " cache " << (c.cache == CacheType::F16 ? "f16" : "f32"));
    const auto reference = alone(*model, sc);
    const auto batched = together(*model, sc);
    REQUIRE(batched.size() == reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
      CHECK(batched[i] == reference[i]);
    }
  }
}

TEST_CASE("a batch may shrink as sequences finish and still match", "[model][batch]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model = load("synthetic-2l-q8_0.gguf", 2, axon::kernels::KernelSet::Auto,
                    axon::model::CacheType::F32, 4);
  const std::size_t n_vocab = model->config().n_vocab;
  // Reference: sequence A gets two steps, sequence B gets four, each alone.
  const auto reference = [&](std::span<const std::int32_t> prompt, std::size_t steps) {
    auto cache = model->new_cache();
    REQUIRE(model->forward_prefill(cache, prompt));
    std::vector<float> out;
    for (std::size_t s = 0; s < steps; ++s) {
      const auto step = model->forward_one(cache, tokens[8 + s]);
      REQUIRE(step);
      out.insert(out.end(), step->begin(), step->end());
    }
    return out;
  };
  const auto ref_a = reference(std::span(tokens).first(2), 2);
  const auto ref_b = reference(std::span(tokens).subspan(2, 5), 4);

  auto ca = model->new_cache();
  auto cb = model->new_cache();
  REQUIRE(model->forward_prefill(ca, std::span(tokens).first(2)));
  REQUIRE(model->forward_prefill(cb, std::span(tokens).subspan(2, 5)));
  std::vector<float> got_a;
  std::vector<float> got_b;
  for (std::size_t s = 0; s < 4; ++s) {
    std::vector<axon::model::KVCache*> caches;
    std::vector<std::int32_t> row;
    if (s < 2) {
      caches.push_back(&ca);
      row.push_back(tokens[8 + s]);
    }
    caches.push_back(&cb);
    row.push_back(tokens[8 + s]);
    const auto logits = model->forward_batch(caches, row);
    REQUIRE(logits);
    std::size_t i = 0;
    if (s < 2) {
      const auto part = logits->subspan(0, n_vocab);
      got_a.insert(got_a.end(), part.begin(), part.end());
      i = 1;
    }
    const auto part = logits->subspan(i * n_vocab, n_vocab);
    got_b.insert(got_b.end(), part.begin(), part.end());
  }
  CHECK(got_a == ref_a);
  CHECK(got_b == ref_b);
}

TEST_CASE("the batched step enforces its bounds and allocates nothing", "[model][batch]") {
  const auto tokens = axon::test::load_axt("synthetic-2l.tokens", "model").as_i32();
  auto model =
      load("synthetic-2l.gguf", 2, axon::kernels::KernelSet::Auto, axon::model::CacheType::F32, 2);
  CHECK(model->max_batch() == 2);
  auto a = model->new_cache();
  auto b = model->new_cache();
  auto c = model->new_cache();
  REQUIRE(model->forward_prefill(a, std::span(tokens).first(3)));
  REQUIRE(model->forward_prefill(b, std::span(tokens).first(1)));
  REQUIRE(model->forward_prefill(c, std::span(tokens).first(2)));
  const std::vector<std::int32_t> two{tokens[5], tokens[6]};
  const std::vector<std::int32_t> three{tokens[5], tokens[6], tokens[7]};

  CHECK_FALSE(model->forward_batch(std::vector<axon::model::KVCache*>{&a, &b, &c},
                                   three));  // over max_batch
  CHECK_FALSE(
      model->forward_batch(std::vector<axon::model::KVCache*>{&a, &b}, three));  // sizes differ
  CHECK_FALSE(model->forward_batch(std::vector<axon::model::KVCache*>{&a, &a}, two));  // twice
  CHECK_FALSE(model->forward_batch(std::vector<axon::model::KVCache*>{},
                                   std::vector<std::int32_t>{}));  // empty
  auto other = axon::model::KVCache(model->config(), 32, axon::model::CacheType::F16);
  CHECK_FALSE(
      model->forward_batch(std::vector<axon::model::KVCache*>{&a, &other}, two));  // wrong type

  const std::vector<axon::model::KVCache*> pair{&a, &b};
  REQUIRE(model->forward_batch(pair, two));
  const auto before = axon::test::allocation_count();
  REQUIRE(model->forward_batch(pair, two));
  CHECK(axon::test::allocation_count() == before);
  CHECK(a.pos() == 5);
  CHECK(b.pos() == 3);

  // A full cache refuses.
  auto full = model->new_cache();
  std::vector<std::int32_t> fill(32);
  for (std::size_t i = 0; i < fill.size(); ++i) {
    fill[i] = tokens[i % tokens.size()];
  }
  for (std::size_t off = 0; off < 32; off += 8) {
    REQUIRE(model->forward_prefill(full, std::span(fill).subspan(off, 8)));
  }
  const std::vector<std::int32_t> one{tokens[0]};
  CHECK_FALSE(model->forward_batch(std::vector<axon::model::KVCache*>{&full}, one));
}
