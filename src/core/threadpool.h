#pragma once

#include "core/function_ref.h"

#include <atomic>
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
class ThreadPool {
public:
  using RangeFn = FunctionRef<void(std::size_t, std::size_t, int)>;

  // n_threads <= 0 selects the number of physical cores.
  explicit ThreadPool(int n_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  [[nodiscard]] int size() const noexcept { return n_threads_; }

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
