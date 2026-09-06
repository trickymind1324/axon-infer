#include "core/threadpool.h"

#include "core/check.h"
#include "core/cpu_info.h"

#include <algorithm>

namespace axon {

namespace {

// Bounded spinning before blocking. Decode issues hundreds of small parallel_for calls per
// token, and waking a sleeping thread through a condition variable costs tens of
// microseconds. These constants are revisited with measurements in the performance phase.
constexpr int kWorkerSpinIterations = 2000;
constexpr int kCallerSpinIterations = 20000;

inline void cpu_relax() {
#if defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

}  // namespace

ThreadPool::ThreadPool(int n_threads) {
  if (n_threads <= 0) {
    n_threads = cpu_info().physical_cores;
  }
  n_threads_ = std::clamp(n_threads, 1, 1024);
  workers_.reserve(static_cast<std::size_t>(n_threads_ - 1));
  for (int i = 1; i < n_threads_; ++i) {
    workers_.emplace_back([this, i] { worker_main(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::scoped_lock lock(mutex_);
    stop_ = true;
    generation_.fetch_add(1, std::memory_order_release);
  }
  wake_cv_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::worker_main(int thread_id) {
  std::uint64_t seen = 0;
  for (;;) {
    // Spin briefly for the next generation, then block.
    bool woke = false;
    for (int i = 0; i < kWorkerSpinIterations; ++i) {
      if (generation_.load(std::memory_order_acquire) != seen) {
        woke = true;
        break;
      }
      cpu_relax();
    }
    if (!woke) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_cv_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
    }
    seen = generation_.load(std::memory_order_acquire);
    {
      // stop_ is written under mutex_ before the generation bump; acquire above orders it.
      const std::scoped_lock lock(mutex_);
      if (stop_) {
        return;
      }
    }

    run_share(thread_id);

    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      const std::scoped_lock lock(mutex_);
      done_cv_.notify_one();
    }
  }
}

void ThreadPool::run_share(int thread_id) {
  const Job& job = job_;
  const std::size_t total = job.end - job.begin;
  try {
    if (job.chunk == 0) {
      // Static split: thread t gets a contiguous slice; the first `extra` slices get one more.
      const auto threads = static_cast<std::size_t>(n_threads_);
      const std::size_t base = total / threads;
      const std::size_t extra = total % threads;
      const auto tid = static_cast<std::size_t>(thread_id);
      const std::size_t sub_begin = job.begin + tid * base + std::min(tid, extra);
      const std::size_t sub_end = sub_begin + base + (tid < extra ? 1 : 0);
      if (sub_begin < sub_end) {
        (*job.fn)(sub_begin, sub_end, thread_id);
      }
      return;
    }
    for (;;) {
      if (failed_.load(std::memory_order_relaxed)) {
        return;
      }
      const std::size_t offset = next_chunk_.fetch_add(job.chunk, std::memory_order_relaxed);
      if (offset >= total) {
        return;
      }
      const std::size_t sub_begin = job.begin + offset;
      const std::size_t sub_end = job.begin + std::min(total, offset + job.chunk);
      (*job.fn)(sub_begin, sub_end, thread_id);
    }
  } catch (...) {
    record_exception();
  }
}

void ThreadPool::record_exception() {
  const std::scoped_lock lock(exception_mutex_);
  if (!exception_) {
    exception_ = std::current_exception();
  }
  failed_.store(true, std::memory_order_relaxed);
}

void ThreadPool::parallel_for(std::size_t begin, std::size_t end, std::size_t chunk, RangeFn fn) {
  if (begin >= end) {
    return;
  }
  AXON_DCHECK(!active_, "parallel_for is not reentrant");
  active_ = true;

  job_ = Job{.begin = begin, .end = end, .chunk = chunk, .fn = &fn};
  next_chunk_.store(0, std::memory_order_relaxed);
  failed_.store(false, std::memory_order_relaxed);
  exception_ = nullptr;

  const int helpers = n_threads_ - 1;
  if (helpers > 0) {
    remaining_.store(helpers, std::memory_order_relaxed);
    {
      const std::scoped_lock lock(mutex_);
      generation_.fetch_add(1, std::memory_order_release);
    }
    wake_cv_.notify_all();
  }

  run_share(0);

  if (helpers > 0) {
    bool done = false;
    for (int i = 0; i < kCallerSpinIterations; ++i) {
      if (remaining_.load(std::memory_order_acquire) == 0) {
        done = true;
        break;
      }
      cpu_relax();
    }
    if (!done) {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
    }
  }

  active_ = false;
  if (failed_.load(std::memory_order_acquire)) {
    std::exception_ptr error;
    {
      const std::scoped_lock lock(exception_mutex_);
      error = exception_;
      exception_ = nullptr;
    }
    failed_.store(false, std::memory_order_relaxed);
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

}  // namespace axon
