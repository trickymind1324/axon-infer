#pragma once

#include "core/function_ref.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace axon {

// Fixed-size pool for data-parallel loops. The calling thread participates as thread 0, so a
// pool of size 1 runs everything inline with no synchronisation. Nothing is allocated per
// call: the job is a FunctionRef plus a few integers stored in the pool.
//
// parallel_for is synchronous and not reentrant: calling it from inside `fn` is a bug.
struct ThreadPoolOptions {
  int n_threads = 0;  // <= 0 selects the number of physical cores
  // How long an idle worker keeps spinning for the next launch before it blocks. Decode
  // issues hundreds of launches per token with single-threaded gaps of tens of microseconds
  // between them, and a worker that has gone to sleep costs a wake-up on every launch.
  std::chrono::microseconds spin_budget{kDefaultSpinBudgetUs};
  // Set from the launch-cost rows in bench_kernels (bm_launch_after_gap) on the M1: with a
  // 30 µs single-threaded gap, a launch costs 0.3 µs at 2 threads and 0.8 µs at 4 with a
  // 200 µs budget, against 4 and 7 µs with no budget. At 8 threads spinning is a net loss
  // (22 µs against 14 µs), because every core is busy and the efficiency cores respond
  // slowly, so the thread count rather than the budget is the lever there.
  static constexpr int kDefaultSpinBudgetUs = 200;
};

class ThreadPool {
public:
  using RangeFn = FunctionRef<void(std::size_t, std::size_t, int)>;

  // n_threads <= 0 selects the number of physical cores.
  explicit ThreadPool(int n_threads = 0);
  explicit ThreadPool(ThreadPoolOptions options);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  [[nodiscard]] int size() const noexcept { return n_threads_; }
  [[nodiscard]] std::chrono::microseconds spin_budget() const noexcept { return spin_budget_; }
  // How many times a worker ran out of spin budget and blocked. For tests and diagnostics.
  [[nodiscard]] std::uint64_t sleeps() const noexcept {
    return sleeps_.load(std::memory_order_relaxed);
  }
  // Number of parallel_for calls so far. The model reports launches per token from it.
  [[nodiscard]] std::uint64_t launches() const noexcept { return launches_; }

  // Invokes fn(sub_begin, sub_end, thread_id) over disjoint sub-ranges that cover
  // [begin, end) exactly once. chunk == 0: static split into size() contiguous pieces.
  // chunk > 0: threads dynamically claim chunk-sized pieces (load balancing).
  // An exception thrown by fn is rethrown on the caller after every thread has stopped.
  void parallel_for(std::size_t begin, std::size_t end, std::size_t chunk, RangeFn fn);

private:
  struct Job {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t chunk = 0;
    RangeFn* fn = nullptr;
  };

  void worker_main(int thread_id);
  void run_share(int thread_id);
  void record_exception();

  int n_threads_;
  std::chrono::microseconds spin_budget_;
  std::atomic<std::uint64_t> sleeps_{0};
  std::uint64_t launches_ = 0;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
  std::atomic<std::uint64_t> generation_{0};
  bool stop_ = false;

  Job job_;
  std::atomic<std::size_t> next_chunk_{0};
  std::atomic<int> remaining_{0};
  std::atomic<bool> failed_{false};
  std::exception_ptr exception_;
  std::mutex exception_mutex_;
  bool active_ = false;
};

}  // namespace axon
