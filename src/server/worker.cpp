#include "server/worker.h"

#include <fmt/format.h>

#include <utility>

namespace axon::server {

Job::Job(CompletionRequest request) : request_(std::move(request)) {}

Job::Poll Job::next(std::string& piece, std::chrono::milliseconds timeout) {
  std::unique_lock lock(mutex_);
  if (!ready_.wait_for(lock, timeout, [&] { return !pieces_.empty() || done_; })) {
    return Poll::Timeout;
  }
  if (!pieces_.empty()) {
    piece = std::move(pieces_.front());
    pieces_.pop_front();
    return Poll::Piece;
  }
  return Poll::Done;
}

void Job::cancel() {
  client_gone_.store(true);
  abort_.store(true);
  // A producer blocked on a full channel must wake up and see the flag.
  ready_.notify_all();
}

bool Job::done() const {
  const std::scoped_lock lock(mutex_);
  return done_;
}

std::string Job::finish_reason() const {
  const std::scoped_lock lock(mutex_);
  return finish_reason_;
}

Usage Job::usage() const {
  const std::scoped_lock lock(mutex_);
  return usage_;
}

std::optional<ApiError> Job::error() const {
  const std::scoped_lock lock(mutex_);
  return error_;
}

void Job::push(std::string piece) {
  const std::scoped_lock lock(mutex_);
  if (client_gone_.load()) {
    return;  // nobody is reading, and the engine stops at the next token anyway
  }
  pieces_.push_back(std::move(piece));
  ready_.notify_all();
}

void Job::finish(std::string finish_reason, Usage usage, std::optional<ApiError> error) {
  const std::scoped_lock lock(mutex_);
  finish_reason_ = std::move(finish_reason);
  usage_ = usage;
  error_ = std::move(error);
  done_ = true;
  ready_.notify_all();
}

Worker::Worker(engine::Engine& engine, std::size_t max_queue)
    : engine_(engine), max_queue_(max_queue), thread_([this] { run(); }) {}

Worker::~Worker() {
  stop();
}

std::shared_ptr<Job> Worker::submit(CompletionRequest request) {
  auto job = std::make_shared<Job>(std::move(request));
  {
    const std::scoped_lock lock(mutex_);
    if (stopping_ || queue_.size() >= max_queue_) {
      return nullptr;
    }
    queue_.push_back(job);
  }
  wake_.notify_one();
  return job;
}

std::size_t Worker::queue_depth() const {
  const std::scoped_lock lock(mutex_);
  return queue_.size();
}

void Worker::stop() {
  {
    const std::scoped_lock lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Worker::run() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;  // stopping with nothing left
      }
      job = std::move(queue_.front());
      queue_.pop_front();
      if (stopping_) {
        job->finish("stop", {},
                    ApiError{.status = 503,
                             .type = "server_error",
                             .code = "shutting_down",
                             .message = "the server is shutting down",
                             .param = std::nullopt});
        continue;
      }
    }
    job->queue_wait_s_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - job->submitted_).count();
    if (job->cancelled()) {
      job->finish("stop", {}, std::nullopt);  // the client left while queued
      continue;
    }
    serve(*job);
    served_.fetch_add(1);
  }
}

void Worker::serve(Job& job) {
  const CompletionRequest& req = job.request();
  engine::GenerateOptions gen = req.gen;
  gen.cancel = job.abort_flag();
  StopMatcher matcher(req.stop);
  const auto out = engine_.generate(req.prompt, gen, [&](std::string_view piece) {
    std::string release = matcher.feed(piece);
    if (!release.empty()) {
      job.push(std::move(release));
    }
    if (matcher.hit()) {
      job.abort();  // a stop string completed, the engine stops after this token
    }
  });
  if (!out) {
    job.finish("stop", {},
               ApiError{.status = 500,
                        .type = "server_error",
                        .code = "generation_failed",
                        .message = out.error().message,
                        .param = std::nullopt});
    return;
  }
  std::string tail = matcher.flush();
  if (!tail.empty()) {
    job.push(std::move(tail));
  }
  const engine::Stats& stats = engine_.stats();
  std::string reason = "stop";
  if (!matcher.hit() && !stats.stopped_on_token && !job.cancelled() &&
      (std::cmp_greater_equal(stats.generated_tokens, req.gen.max_tokens) || stats.context_full)) {
    reason = "length";
  }
  job.finish(
      reason,
      Usage{.prompt_tokens = stats.prompt_tokens, .completion_tokens = stats.generated_tokens},
      std::nullopt);
}

}  // namespace axon::server
