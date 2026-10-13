#include "engine/bench.h"

#include "engine/engine.h"
#include "kernels/kernels.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

namespace axon::engine {

namespace {

double load_average() {
  double loads[1] = {0.0};
  return getloadavg(loads, 1) == 1 ? loads[0] : 0.0;
}

double median(std::vector<double> values) {
  std::ranges::sort(values);
  const std::size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

}  // namespace

Expected<BenchResult> run_benchmark(model::LlamaModel& model, std::span<const std::int32_t> prompt,
                                    const BenchOptions& opts,
                                    const std::function<void(const BenchRun&, int)>& progress) {
  if (opts.generate < 1 || opts.runs < 1) {
    return make_error(ErrorCode::InvalidArgument,
                      "benchmark: generate and runs must be at least 1");
  }
  if (prompt.empty()) {
    return make_error(ErrorCode::InvalidArgument, "benchmark: empty prompt");
  }
  const auto needed = prompt.size() + static_cast<std::size_t>(opts.generate);
  if (needed > model.max_ctx()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("benchmark: {} prompt plus {} generated tokens exceed the "
                                  "context of {}",
                                  prompt.size(), opts.generate, model.max_ctx()));
  }
  BenchResult result;
  result.prompt_tokens = prompt.size();
  result.generated_tokens = static_cast<std::size_t>(opts.generate);
  result.threads = model.n_threads();
  result.kernels = std::string(kernels::name(model.table()));
  result.cache = model.cache_type() == model::CacheType::F16 ? "f16" : "f32";
  result.context = model.max_ctx();
  result.spin_budget_us = static_cast<int>(model.pool().spin_budget().count());
  result.load_average_before = load_average();

  GenerateOptions gen;
  gen.max_tokens = opts.generate;
  gen.sampler.temperature = 0.0F;
  gen.sampler.repeat_penalty = 1.0F;
  gen.stop_on_tokens = false;
  Engine engine(model);
  for (int run = -1; run < opts.runs; ++run) {
    engine.clear();  // a fresh cache every run, so prefill is always measured
    auto out = engine.generate(prompt, gen, nullptr);
    if (!out) {
      return out.error();
    }
    const Stats& s = engine.stats();
    if (std::cmp_not_equal(s.generated_tokens, opts.generate)) {
      return make_error(ErrorCode::Internal,
                        fmt::format("benchmark: generated {} tokens instead of {}",
                                    s.generated_tokens, opts.generate));
    }
    if (run < 0) {
      continue;  // warm-up
    }
    const BenchRun measured{
        .prefill_s = s.prefill_s, .decode_s = s.decode_s, .peak_rss_mb = s.peak_rss_mb};
    result.runs.push_back(measured);
    if (progress) {
      progress(measured, run);
    }
  }
  result.load_average_after = load_average();
  {
    // One more decoded token on its own, to count the launches a decode step issues.
    engine.clear();
    GenerateOptions one = gen;
    one.max_tokens = 2;
    const auto before = model.pool().launches();
    const auto out = engine.generate(prompt.first(1), one, nullptr);
    if (out) {
      // Two forwards ran: the prompt token and one decode step. Both are single-position
      // passes, so half the launches is one token's worth.
      result.launches_per_token = static_cast<double>(model.pool().launches() - before) / 2.0;
    }
  }

  std::vector<double> prefill;
  std::vector<double> decode;
  for (const auto& r : result.runs) {
    prefill.push_back(static_cast<double>(result.prompt_tokens) / r.prefill_s);
    decode.push_back(static_cast<double>(result.generated_tokens) / r.decode_s);
    result.peak_rss_mb = std::max(result.peak_rss_mb, r.peak_rss_mb);
  }
  result.prefill_tok_s_median = median(prefill);
  result.prefill_tok_s_best = *std::ranges::max_element(prefill);
  result.decode_tok_s_median = median(decode);
  result.decode_tok_s_best = *std::ranges::max_element(decode);
  result.weight_bytes_per_token = model.weight_bytes_per_token();
  const auto& cfg = model.config();
  const double mean_position = static_cast<double>(result.prompt_tokens) +
                               static_cast<double>(result.generated_tokens - 1) / 2.0;
  const double element_bytes = model.cache_type() == model::CacheType::F16 ? 2.0 : 4.0;
  result.cache_bytes_per_token =
      2.0 * cfg.n_layer * static_cast<double>(cfg.n_embd_kv()) * element_bytes * mean_position;
  result.weight_gb_s_median =
      static_cast<double>(result.weight_bytes_per_token) * result.decode_tok_s_median / 1e9;
  return result;
}

namespace {

model::ForwardOptions profiled(model::ForwardProfile* profile) {
  model::ForwardOptions opts;
  opts.profile = profile;
  return opts;
}

Expected<bool> prefill_chunks(model::LlamaModel& model, model::KVCache& cache,
                              std::span<const std::int32_t> prompt,
                              model::ForwardProfile* profile) {
  for (std::size_t offset = 0; offset < prompt.size(); offset += model.max_seq()) {
    const std::size_t n = std::min(model.max_seq(), prompt.size() - offset);
    const auto out = model.forward_prefill(cache, prompt.subspan(offset, n), profiled(profile));
    if (!out) {
      return out.error();
    }
  }
  return true;
}

}  // namespace

Expected<ProfileReport> profile_forward(model::LlamaModel& model,
                                        std::span<const std::int32_t> prompt) {
  if (prompt.empty()) {
    return make_error(ErrorCode::InvalidArgument, "profile: the prompt is empty");
  }
  if (prompt.size() + 1 > model.max_ctx()) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("profile: the prompt and one token need {} positions, "
                                  "the context has {}",
                                  prompt.size() + 1, model.max_ctx()));
  }
  ProfileReport report;
  report.prompt_tokens = prompt.size();
  const std::int32_t next = prompt.back();
  {
    auto warm = model.new_cache();
    const auto filled = prefill_chunks(model, warm, prompt, nullptr);
    if (!filled) {
      return filled.error();
    }
    const auto one = model.forward_one(warm, next);
    if (!one) {
      return one.error();
    }
  }
  auto cache = model.new_cache();
  const auto filled = prefill_chunks(model, cache, prompt, &report.prefill);
  if (!filled) {
    return filled.error();
  }
  const auto one = model.forward_prefill(cache, std::span(&next, 1), profiled(&report.decode));
  if (!one) {
    return one.error();
  }
  return report;
}

std::string profile_text(const ProfileReport& report) {
  using model::ForwardProfile;
  const double prefill_total = report.prefill.total();
  const double decode_total = report.decode.total();
  const auto per_prompt_token = static_cast<double>(report.prompt_tokens);
  std::string out = fmt::format("{:<10} {:>14} {:>7} {:>14} {:>7}\n", "launch", "prefill ms/tok",
                                "share", "decode ms/tok", "share");
  for (std::size_t i = 0; i < ForwardProfile::kSlots; ++i) {
    const auto slot = static_cast<ForwardProfile::Slot>(i);
    const double p = report.prefill.seconds[i];
    const double d = report.decode.seconds[i];
    out += fmt::format("{:<10} {:>14.3f} {:>6.1f}% {:>14.3f} {:>6.1f}%\n",
                       ForwardProfile::name(slot), 1e3 * p / per_prompt_token,
                       prefill_total > 0.0 ? 100.0 * p / prefill_total : 0.0, 1e3 * d,
                       decode_total > 0.0 ? 100.0 * d / decode_total : 0.0);
  }
  out += fmt::format("{:<10} {:>14.3f} {:>7} {:>14.3f} {:>7}\n", "total",
                     1e3 * prefill_total / per_prompt_token, "", 1e3 * decode_total, "");
  return out;
}

std::string bench_json(const BenchResult& r, std::string_view model_file,
                       std::string_view description) {
  std::string runs;
  for (std::size_t i = 0; i < r.runs.size(); ++i) {
    runs += fmt::format(
        R"json({}{{"prefill_s": {:.4f}, "decode_s": {:.4f}, "peak_rss_mb": {:.1f}}})json",
        i == 0 ? "" : ", ", r.runs[i].prefill_s, r.runs[i].decode_s, r.runs[i].peak_rss_mb);
  }
  return fmt::format(
      R"json({{"model_file": "{}", "model": "{}", "kernels": "{}", "cache": "{}", "threads": {}, "context": {}, "spin_budget_us": {}, "launches_per_token": {:.1f}, "prompt_tokens": {}, "generated_tokens": {}, "runs": {}, "prefill_tok_s": {{"median": {:.3f}, "best": {:.3f}}}, "decode_tok_s": {{"median": {:.3f}, "best": {:.3f}}}, "weight_bytes_per_token": {}, "cache_bytes_per_token": {:.0f}, "weight_gb_s": {:.3f}, "peak_rss_mb": {:.1f}, "load_average": {{"before": {:.2f}, "after": {:.2f}}}, "measurements": [{}]}})json",
      model_file, description, r.kernels, r.cache, r.threads, r.context, r.spin_budget_us,
      r.launches_per_token, r.prompt_tokens, r.generated_tokens, r.runs.size(),
      r.prefill_tok_s_median, r.prefill_tok_s_best, r.decode_tok_s_median, r.decode_tok_s_best,
      r.weight_bytes_per_token, r.cache_bytes_per_token, r.weight_gb_s_median, r.peak_rss_mb,
      r.load_average_before, r.load_average_after, runs);
}

}  // namespace axon::engine
