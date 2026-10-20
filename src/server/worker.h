#pragma once

// One thread owning the engine, serving requests in arrival order. A request becomes a Job:
// the worker pushes decoded text into it, the HTTP handler pops, and either side can end it.
// The handler cancels when its client has gone, the worker finishes when generation ends.

#include "engine/engine.h"
#include "server/api.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace axon::server {

class Job {
public:
  explicit Job(CompletionRequest request);

  enum class Poll : std::uint8_t { Piece, Done, Timeout };
  // Consumer side. Waits up to `timeout` for the next piece of text. Done means generation has
  // ended and everything was delivered, so finish_reason(), usage() and error() are final.
  Poll next(std::string& piece, std::chrono::milliseconds timeout);
  // Consumer side. Ends generation after the token in flight and releases the worker.
  void cancel();

  [[nodiscard]] const CompletionRequest& request() const noexcept { return request_; }
  [[nodiscard]] bool done() const;
  [[nodiscard]] bool cancelled() const noexcept { return client_gone_.load(); }
  [[nodiscard]] std::string finish_reason() const;  // "stop" or "length" once done
  [[nodiscard]] Usage usage() const;
  [[nodiscard]] std::optional<ApiError> error() const;
  [[nodiscard]] double queue_wait_s() const noexcept { return queue_wait_s_; }

private:
  friend class Worker;
  // Producer side.
  void push(std::string piece);
  void finish(std::string finish_reason, Usage usage, std::optional<ApiError> error);
  // The flag the engine polls: set by cancel() and by a completed stop string.
  [[nodiscard]] const std::atomic<bool>* abort_flag() const noexcept { return &abort_; }
  void abort() noexcept { abort_.store(true); }

  CompletionRequest request_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::string> pieces_;
  bool done_ = false;
  std::string finish_reason_;
  Usage usage_;
  std::optional<ApiError> error_;
  std::atomic<bool> abort_{false};
  std::atomic<bool> client_gone_{false};
  std::chrono::steady_clock::time_point submitted_ = std::chrono::steady_clock::now();
  double queue_wait_s_ = 0.0;
};

class Worker {
public:
  Worker(engine::Engine& engine, std::size_t max_queue);
  ~Worker();
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Queues a request. Null when the queue is full, which the caller turns into 503.
  std::shared_ptr<Job> submit(CompletionRequest request);
  [[nodiscard]] std::size_t queue_depth() const;
  [[nodiscard]] std::uint64_t served() const noexcept { return served_.load(); }
  // Ends the thread after the job in flight. Queued jobs finish with an error.
  void stop();

private:
  void run();
  void serve(Job& job);

  engine::Engine& engine_;
  std::size_t max_queue_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> queue_;
  bool stopping_ = false;
  std::atomic<std::uint64_t> served_{0};
  std::thread thread_;
};

}  // namespace axon::server
