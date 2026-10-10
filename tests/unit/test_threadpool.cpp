#include "core/threadpool.h"

#include "alloc_counter.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("parallel_for visits every index exactly once", "[core][threadpool]") {
  const int threads = GENERATE(1, 2, 3, 8);
  const auto [begin, end] = GENERATE(
      std::pair<std::size_t, std::size_t>{0, 0}, std::pair<std::size_t, std::size_t>{0, 1},
      std::pair<std::size_t, std::size_t>{0, 7}, std::pair<std::size_t, std::size_t>{0, 1000},
      std::pair<std::size_t, std::size_t>{5, 1003});
  const std::size_t chunk =
      GENERATE(std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{64}, std::size_t{5000});

  axon::ThreadPool pool(threads);
  REQUIRE(pool.size() == threads);

  std::vector<std::atomic<int>> hits(end);
  std::atomic<int> bad_thread_ids{0};
  pool.parallel_for(begin, end, chunk, [&](std::size_t b, std::size_t e, int tid) {
    if (tid < 0 || tid >= threads) {
      bad_thread_ids.fetch_add(1);
    }
    for (std::size_t i = b; i < e; ++i) {
      hits[i].fetch_add(1);
    }
  });

  CHECK(bad_thread_ids.load() == 0);
  for (std::size_t i = 0; i < begin; ++i) {
    CHECK(hits[i].load() == 0);
  }
  for (std::size_t i = begin; i < end; ++i) {
    CHECK(hits[i].load() == 1);
  }
}

TEST_CASE("parallel_for with an empty or inverted range never calls fn", "[core][threadpool]") {
  axon::ThreadPool pool(4);
  std::atomic<int> calls{0};
  pool.parallel_for(10, 10, 0, [&](std::size_t, std::size_t, int) { calls.fetch_add(1); });
  pool.parallel_for(10, 3, 1, [&](std::size_t, std::size_t, int) { calls.fetch_add(1); });
  CHECK(calls.load() == 0);
}

TEST_CASE("static split issues at most size() contiguous calls", "[core][threadpool]") {
  axon::ThreadPool pool(4);
  std::atomic<int> calls{0};
  std::atomic<std::size_t> covered{0};
  pool.parallel_for(0, 10, 0, [&](std::size_t b, std::size_t e, int) {
    calls.fetch_add(1);
    covered.fetch_add(e - b);
  });
  CHECK(calls.load() <= 4);
  CHECK(covered.load() == 10);
}

TEST_CASE("parallel_for survives many rapid invocations", "[core][threadpool]") {
  axon::ThreadPool pool(4);
  std::atomic<std::size_t> total{0};
  for (int iteration = 0; iteration < 10000; ++iteration) {
    pool.parallel_for(0, 8, 1, [&](std::size_t b, std::size_t e, int) {
      total.fetch_add(e - b, std::memory_order_relaxed);
    });
  }
  CHECK(total.load() == std::size_t{8} * 10000);
}

TEST_CASE("parallel_for does not allocate per call", "[core][threadpool]") {
  axon::ThreadPool pool(3);
  std::atomic<std::size_t> sink{0};
  auto body = [&](std::size_t b, std::size_t e, int) { sink.fetch_add(e - b); };
  pool.parallel_for(0, 100, 10, body);  // warm up
  const std::size_t before = axon::test::allocation_count();
  for (int i = 0; i < 100; ++i) {
    pool.parallel_for(0, 100, 10, body);
  }
  CHECK(axon::test::allocation_count() == before);
  CHECK(sink.load() == std::size_t{101} * 100);
}

TEST_CASE("exception in a range is rethrown on the caller", "[core][threadpool]") {
  axon::ThreadPool pool(4);
  std::atomic<int> completed{0};
  REQUIRE_THROWS_AS(pool.parallel_for(0, 64, 1,
                                      [&](std::size_t b, std::size_t, int) {
                                        if (b == 17) {
                                          throw std::runtime_error("boom at 17");
                                        }
                                        completed.fetch_add(1);
                                      }),
                    std::runtime_error);

  // The pool is still usable afterwards.
  std::atomic<std::size_t> total{0};
  pool.parallel_for(0, 100, 7, [&](std::size_t b, std::size_t e, int) { total.fetch_add(e - b); });
  CHECK(total.load() == 100);
}

TEST_CASE("exception message is preserved", "[core][threadpool]") {
  axon::ThreadPool pool(2);
  try {
    pool.parallel_for(0, 2, 0, [&](std::size_t, std::size_t, int tid) {
      if (tid == 1) {
        throw std::runtime_error("worker failure");
      }
    });
    FAIL("expected an exception");
  } catch (const std::runtime_error& error) {
    CHECK(std::string(error.what()) == "worker failure");
  }
}

TEST_CASE("default thread count is positive", "[core][threadpool]") {
  const axon::ThreadPool pool(0);
  CHECK(pool.size() >= 1);
}

TEST_CASE("workers spin for the budget and then sleep", "[core][threadpool][spin]") {
  // A zero budget sleeps after every launch that leaves a gap.
  axon::ThreadPool eager({.n_threads = 3, .spin_budget = std::chrono::microseconds(0)});
  CHECK(eager.spin_budget().count() == 0);
  std::atomic<int> hits{0};
  for (int i = 0; i < 5; ++i) {
    eager.parallel_for(0, 3, 0, [&](std::size_t, std::size_t, int) { hits.fetch_add(1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  CHECK(hits.load() == 15);
  CHECK(eager.launches() == 5);
  CHECK(eager.sleeps() >= 8);  // two workers, at least four gaps each

  // A generous budget covers short gaps without a single sleep.
  axon::ThreadPool patient({.n_threads = 3, .spin_budget = std::chrono::milliseconds(50)});
  for (int i = 0; i < 20; ++i) {
    patient.parallel_for(0, 3, 0, [&](std::size_t, std::size_t, int) { hits.fetch_add(1); });
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
    while (std::chrono::steady_clock::now() < until) {
    }
  }
  CHECK(hits.load() == 75);
  CHECK(patient.sleeps() == 0);
  // Past the budget the workers do go to sleep, so an idle process frees its cores.
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  CHECK(patient.sleeps() == 2);
  patient.parallel_for(0, 3, 0, [&](std::size_t, std::size_t, int) { hits.fetch_add(1); });
  CHECK(hits.load() == 78);
}
