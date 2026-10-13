#pragma once

// End-to-end benchmark: prefill a fixed prompt, decode a fixed number of tokens, repeat, and
// report medians together with the bytes the run moved, so a throughput number can be read
// against the machine's memory bandwidth.

#include "core/expected.h"

#include "model/llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace axon::engine {

struct BenchOptions {
  int generate = 128;  // tokens decoded per run, always exactly this many
  int runs = 5;        // measured runs after one warm-up
};

struct BenchRun {
  double prefill_s = 0.0;
  double decode_s = 0.0;
  double peak_rss_mb = 0.0;
};

struct BenchResult {
  std::size_t prompt_tokens = 0;
  std::size_t generated_tokens = 0;
  int threads = 0;
  std::string kernels;  // table name, "scalar" or "neon"
  std::string cache;    // "f32" or "f16"
  std::size_t context = 0;
  int spin_budget_us = 0;
  double launches_per_token = 0.0;  // parallel launches per decoded token
  std::vector<BenchRun> runs;       // the measured runs, warm-up excluded
  double prefill_tok_s_median = 0.0;
  double prefill_tok_s_best = 0.0;
  double decode_tok_s_median = 0.0;
  double decode_tok_s_best = 0.0;
  double peak_rss_mb = 0.0;
  std::size_t weight_bytes_per_token = 0;  // every weight byte one forward pass reads
  double cache_bytes_per_token = 0.0;      // keys and values read per decoded token, averaged
  double weight_gb_s_median = 0.0;         // weight bytes times median decode tokens per second
  double load_average_before = 0.0;        // one-minute load average, to flag noisy runs
  double load_average_after = 0.0;
};

// Where the forward pass spends its time, by launch kind: the whole prompt prefilled in
// max_seq chunks on a fresh cache, then one decoded token. One unprofiled pass runs first so
// page faults on the weights are not charged to a slot.
struct ProfileReport {
  model::ForwardProfile prefill;
  model::ForwardProfile decode;
  std::size_t prompt_tokens = 0;
};
Expected<ProfileReport> profile_forward(model::LlamaModel& model,
                                        std::span<const std::int32_t> prompt);
// One line per slot: milliseconds per prompt token and per decoded token with their shares.
std::string profile_text(const ProfileReport& report);

// One warm-up plus `runs` measured runs of prefill-then-decode on a fresh cache each time,
// greedy, never stopping on end-of-generation tokens. The prompt must fit the context with
// room for the generated tokens. Progress is called after each measured run.
Expected<BenchResult> run_benchmark(model::LlamaModel& model, std::span<const std::int32_t> prompt,
                                    const BenchOptions& opts,
                                    const std::function<void(const BenchRun&, int)>& progress = {});

// The result as one JSON object. `model_file` and `description` are copied in as strings.
std::string bench_json(const BenchResult& result, std::string_view model_file,
                       std::string_view description);

}  // namespace axon::engine
