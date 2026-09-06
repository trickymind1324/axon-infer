#include "core/threadpool.h"

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>

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

}  // namespace

BENCHMARK(bm_parallel_for_overhead)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseRealTime();
