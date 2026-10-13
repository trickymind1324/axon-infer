#include "core/dtype.h"
#include "core/threadpool.h"

#include "kernels/kernels.h"

#include <benchmark/benchmark.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Cost of one parallel_for round trip with trivial work. This is the fixed overhead paid by
// every kernel launch during decode, so it is tracked from the start.
void bm_parallel_for_overhead(benchmark::State& state) {
  const auto threads = static_cast<int>(state.range(0));
  axon::ThreadPool pool(threads);
  std::atomic<std::size_t> sink{0};
  for (auto _ : state) {
    pool.parallel_for(0, static_cast<std::size_t>(threads), 0,
                      [&](std::size_t begin, std::size_t end, int) {
                        sink.fetch_add(end - begin, std::memory_order_relaxed);
                      });
  }
  benchmark::DoNotOptimize(sink.load());
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));
}

std::vector<float> random_floats(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> out(n);
  for (auto& v : out) {
    v = dist(rng);
  }
  return out;
}

// Reports GFLOP/s (two flops per weight) and GB/s of weights read, which is the number that
// matters: a decode step is one pass over every weight in the model.
void report_matmul(benchmark::State& state, std::size_t n_out, std::size_t n_in,
                   std::size_t weight_bytes) {
  const auto calls = static_cast<double>(state.iterations());
  state.counters["GFLOP/s"] =
      benchmark::Counter(calls * 2.0 * static_cast<double>(n_out * n_in),
                         benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
  state.counters["GB/s"] =
      benchmark::Counter(calls * static_cast<double>(weight_bytes), benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1000);
}

// Single-vector matmul at the shapes of the SmolLM2-135M (576 x 1536) and 2B-class
// (2048 x 2048, 2048 x 8192) projections. Args are {n_out, n_in}.
// Args are {n_out, n_in, table}: table 0 is scalar and 1 the best the host supports.
const axon::kernels::KernelTable& table_for(benchmark::State& state) {
  if (state.range(2) == 0) {
    return axon::kernels::scalar_kernels();
  }
  return **axon::kernels::select(axon::kernels::KernelSet::Auto);
}

void bm_matmul_f32(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const auto w = random_floats(n_out * n_in, 1);
  const auto x = random_floats(n_in, 2);
  std::vector<float> y(n_out);
  const auto& k = table_for(state);
  for (auto _ : state) {
    k.matmul_f32(w.data(), x.data(), y.data(), n_out, n_in);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul(state, n_out, n_in, w.size() * sizeof(float));
}

void bm_matmul_f16(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const auto w_f32 = random_floats(n_out * n_in, 1);
  std::vector<std::uint16_t> w(w_f32.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    w[i] = axon::fp32_to_fp16(w_f32[i]);
  }
  const auto x = random_floats(n_in, 2);
  std::vector<float> y(n_out);
  const auto& k = table_for(state);
  for (auto _ : state) {
    k.matmul_f16(w.data(), x.data(), y.data(), n_out, n_in);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul(state, n_out, n_in, w.size() * sizeof(std::uint16_t));
}

// Q8_0 weights against float activations (dequantize on the fly) and against Q8_0
// activations (integer dot). The second is the path the model uses by default. Args are
// {n_out, n_in, int_dot, table}, table 0 scalar and 1 the best the host supports.
void bm_matmul_q8_0(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const bool int_dot = state.range(2) != 0;
  const auto& k = state.range(3) == 0 ? axon::kernels::scalar_kernels()
                                      : **axon::kernels::select(axon::kernels::KernelSet::Auto);
  const auto w = random_floats(n_out * n_in, 1);
  std::vector<axon::BlockQ8_0> wq(n_out * n_in / 32);
  axon::quantize_row_q8_0(w.data(), wq.data(), w.size());
  const auto x = random_floats(n_in, 2);
  std::vector<axon::BlockQ8_0> xq(n_in / 32);
  axon::quantize_row_q8_0(x.data(), xq.data(), n_in);
  std::vector<float> y(n_out);
  for (auto _ : state) {
    if (int_dot) {
      k.matmul_q8_0_q8(wq.data(), xq.data(), y.data(), n_out, n_in);
    } else {
      k.matmul_q8_0(wq.data(), x.data(), y.data(), n_out, n_in);
    }
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul(state, n_out, n_in, wq.size() * sizeof(axon::BlockQ8_0));
}

// Q4_0 weights are built by hand from the same floats: the nibble layout is fixed, the exact
// values do not matter for timing.
void bm_matmul_q4_0(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const bool int_dot = state.range(2) != 0;
  const auto& k = state.range(3) == 0 ? axon::kernels::scalar_kernels()
                                      : **axon::kernels::select(axon::kernels::KernelSet::Auto);
  std::vector<axon::BlockQ4_0> wq(n_out * n_in / 32);
  std::uint32_t pattern = 0x9E3779B9U;
  for (auto& block : wq) {
    block.d = axon::fp32_to_fp16(0.01F);
    for (auto& q : block.qs) {
      pattern = pattern * 1664525U + 1013904223U;  // any spread of nibbles will do for timing
      q = static_cast<std::uint8_t>(pattern >> 24);
    }
  }
  const auto x = random_floats(n_in, 2);
  std::vector<axon::BlockQ8_0> xq(n_in / 32);
  axon::quantize_row_q8_0(x.data(), xq.data(), n_in);
  std::vector<float> y(n_out);
  for (auto _ : state) {
    if (int_dot) {
      k.matmul_q4_0_q8(wq.data(), xq.data(), y.data(), n_out, n_in);
    } else {
      k.matmul_q4_0(wq.data(), x.data(), y.data(), n_out, n_in);
    }
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul(state, n_out, n_in, wq.size() * sizeof(axon::BlockQ4_0));
}

// Q6_K weights against float activations. Args are {n_out, n_in, table}.
void bm_matmul_q6_k(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  std::vector<axon::BlockQ6_K> wq(n_out * n_in / 256);
  std::uint32_t pattern = 0x2545F491U;
  for (auto& block : wq) {
    for (auto& q : block.ql) {
      pattern = pattern * 1664525U + 1013904223U;
      q = static_cast<std::uint8_t>(pattern >> 24);
    }
    for (auto& q : block.qh) {
      pattern = pattern * 1664525U + 1013904223U;
      q = static_cast<std::uint8_t>(pattern >> 24);
    }
    for (auto& sc : block.scales) {
      pattern = pattern * 1664525U + 1013904223U;
      sc = static_cast<std::int8_t>(pattern >> 25);
    }
    block.d = axon::fp32_to_fp16(0.001F);
  }
  const auto x = random_floats(n_in, 2);
  std::vector<float> y(n_out);
  const auto& k = table_for(state);
  for (auto _ : state) {
    k.matmul_q6_k(wq.data(), x.data(), y.data(), n_out, n_in);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul(state, n_out, n_in, wq.size() * sizeof(axon::BlockQ6_K));
}

// Launch cost as the model sees it: a single-threaded gap of `gap_us` between launches, like
// the RMSNorm and RoPE work between matmuls, then one trivial parallel_for. Args are
// {threads, spin budget us, gap us}. The reported time minus the gap is the launch cost.
// Multi-row matmul at the prompt-sized row count the model's prefill chunks use, reported per
// weight-row pass so the GFLOP/s column reads against the single-row rows above. Args are
// {n_out, n_in, table}.
constexpr std::size_t kBenchRows = 32;

void report_matmul_rows(benchmark::State& state, std::size_t n_out, std::size_t n_in,
                        std::size_t weight_bytes) {
  const auto calls = static_cast<double>(state.iterations());
  state.counters["GFLOP/s"] =
      benchmark::Counter(calls * 2.0 * static_cast<double>(kBenchRows * n_out * n_in),
                         benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
  state.counters["GB/s"] =
      benchmark::Counter(calls * static_cast<double>(weight_bytes), benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1000);
}

std::vector<axon::BlockQ6_K> random_q6_k(std::size_t n_blocks) {
  std::vector<axon::BlockQ6_K> wq(n_blocks);
  std::uint32_t pattern = 0x2545F491U;
  for (auto& block : wq) {
    for (auto& q : block.ql) {
      pattern = pattern * 1664525U + 1013904223U;
      q = static_cast<std::uint8_t>(pattern >> 24);
    }
    for (auto& q : block.qh) {
      pattern = pattern * 1664525U + 1013904223U;
      q = static_cast<std::uint8_t>(pattern >> 24);
    }
    for (auto& sc : block.scales) {
      pattern = pattern * 1664525U + 1013904223U;
      sc = static_cast<std::int8_t>(pattern >> 25);
    }
    block.d = axon::fp32_to_fp16(0.001F);
  }
  return wq;
}

// Q8_0 and Q4_0 weights with Q8_0 activations over kBenchRows rows. Args are {n_out, n_in,
// four_bit, table}.
void bm_matmul_int_rows(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const bool four_bit = state.range(2) != 0;
  const auto& k = state.range(3) == 0 ? axon::kernels::scalar_kernels()
                                      : **axon::kernels::select(axon::kernels::KernelSet::Auto);
  const auto w = random_floats(n_out * n_in, 1);
  std::vector<axon::BlockQ8_0> w8(n_out * n_in / 32);
  axon::quantize_row_q8_0(w.data(), w8.data(), w.size());
  std::vector<axon::BlockQ4_0> w4(w8.size());
  for (std::size_t b = 0; b < w4.size(); ++b) {
    w4[b].d = w8[b].d;
    for (std::size_t i = 0; i < 16; ++i) {
      w4[b].qs[i] = static_cast<std::uint8_t>((w8[b].qs[i] & 0x0F) | (w8[b].qs[i + 16] << 4));
    }
  }
  const auto x = random_floats(kBenchRows * n_in, 2);
  std::vector<axon::BlockQ8_0> xq(kBenchRows * n_in / 32);
  for (std::size_t r = 0; r < kBenchRows; ++r) {
    axon::quantize_row_q8_0(x.data() + r * n_in, xq.data() + r * (n_in / 32), n_in);
  }
  std::vector<float> y(kBenchRows * n_out);
  for (auto _ : state) {
    if (four_bit) {
      k.matmul_q4_0_q8_rows(w4.data(), xq.data(), y.data(), kBenchRows, n_out, n_in, n_out);
    } else {
      k.matmul_q8_0_q8_rows(w8.data(), xq.data(), y.data(), kBenchRows, n_out, n_in, n_out);
    }
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul_rows(state, n_out, n_in,
                     four_bit ? w4.size() * sizeof(axon::BlockQ4_0)
                              : w8.size() * sizeof(axon::BlockQ8_0));
}

void bm_matmul_q6_k_rows(benchmark::State& state) {
  const auto n_out = static_cast<std::size_t>(state.range(0));
  const auto n_in = static_cast<std::size_t>(state.range(1));
  const auto wq = random_q6_k(n_out * n_in / 256);
  const auto x = random_floats(kBenchRows * n_in, 2);
  std::vector<float> y(kBenchRows * n_out);
  const auto& k = table_for(state);
  for (auto _ : state) {
    k.matmul_q6_k_rows(wq.data(), x.data(), y.data(), kBenchRows, n_out, n_in, n_out);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  report_matmul_rows(state, n_out, n_in, wq.size() * sizeof(axon::BlockQ6_K));
}

void bm_launch_after_gap(benchmark::State& state) {
  const auto threads = static_cast<int>(state.range(0));
  const auto budget = static_cast<int>(state.range(1));
  const auto gap = std::chrono::microseconds(state.range(2));
  axon::ThreadPool pool({.n_threads = threads, .spin_budget = std::chrono::microseconds(budget)});
  std::atomic<std::size_t> sink{0};
  for (auto _ : state) {
    const auto until = std::chrono::steady_clock::now() + gap;
    while (std::chrono::steady_clock::now() < until) {
    }
    pool.parallel_for(0, static_cast<std::size_t>(threads), 0,
                      [&](std::size_t begin, std::size_t end, int) {
                        sink.fetch_add(end - begin, std::memory_order_relaxed);
                      });
  }
  benchmark::DoNotOptimize(sink.load());
  state.counters["sleeps"] = static_cast<double>(pool.sleeps());
  state.counters["launch_us"] = benchmark::Counter((static_cast<double>(state.iterations()) * 0.0),
                                                   benchmark::Counter::kAvgIterations);
}

}  // namespace

BENCHMARK(bm_parallel_for_overhead)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime();
BENCHMARK(bm_matmul_f32)
    ->Args({1536, 576, 0})
    ->Args({2048, 2048, 0})
    ->Args({8192, 2048, 0})
    ->Args({1536, 576, 1})
    ->Args({2048, 2048, 1})
    ->Args({8192, 2048, 1});
BENCHMARK(bm_matmul_f16)
    ->Args({1536, 576, 0})
    ->Args({2048, 2048, 0})
    ->Args({8192, 2048, 0})
    ->Args({1536, 576, 1})
    ->Args({2048, 2048, 1})
    ->Args({8192, 2048, 1});
BENCHMARK(bm_matmul_q8_0)
    ->Args({2048, 2048, 0, 0})
    ->Args({2048, 2048, 1, 0})
    ->Args({8192, 2048, 1, 0})
    ->Args({2048, 2048, 1, 1})
    ->Args({8192, 2048, 1, 1});
BENCHMARK(bm_matmul_q4_0)
    ->Args({2048, 2048, 0, 0})
    ->Args({2048, 2048, 1, 0})
    ->Args({8192, 2048, 1, 0})
    ->Args({2048, 2048, 1, 1})
    ->Args({8192, 2048, 1, 1});
BENCHMARK(bm_matmul_q6_k)
    ->Args({2048, 2048, 0})
    ->Args({8192, 2048, 0})
    ->Args({2048, 2048, 1})
    ->Args({8192, 2048, 1});
BENCHMARK(bm_matmul_q6_k_rows)->Args({2048, 2048, 0})->Args({2048, 2048, 1});
BENCHMARK(bm_matmul_int_rows)
    ->Args({2048, 2048, 0, 0})
    ->Args({2048, 2048, 0, 1})
    ->Args({2048, 2048, 1, 0})
    ->Args({2048, 2048, 1, 1});
BENCHMARK(bm_launch_after_gap)
    ->ArgsProduct({{2, 4, 8}, {0, 50, 200, 1000}, {30}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
