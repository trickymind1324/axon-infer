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

Worker::Worker(engine::BatchEngine& batch, std::size_t max_queue)
    : batch_(batch), max_queue_(max_queue), active_(batch.slots()), thread_([this] { run(); }) {}

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

std::size_t Worker::running() const {
  return running_.load();
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

void Worker::fail(Job& job, std::string message) {
  job.finish("stop", {},
             ApiError{.status = 500,
                      .type = "server_error",
                      .code = "generation_failed",
                      .message = std::move(message),
                      .param = std::nullopt});
}

void Worker::admit(std::shared_ptr<Job> job) {
  job->queue_wait_s_ =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - job->submitted_).count();
  if (job->cancelled()) {
    job->finish("stop", {}, std::nullopt);  // the client left while queued
    return;
  }
  const CompletionRequest& req = job->request();
  engine::GenerateOptions gen = req.gen;
  gen.cancel = job->abort_flag();
  auto matcher = std::make_unique<StopMatcher>(req.stop);
  // The slot is known only once admitted, so the sinks look it up through `this`.
  const auto slot = batch_.admit(
      req.prompt, gen,
      [this, job](std::string_view piece) {
        for (engine::SlotId s = 0; s < active_.size(); ++s) {
          if (active_[s].job == job) {
            on_text(s, piece);
            return;
          }
        }
      },
      [this, job](engine::Finish finish, const engine::Stats& stats) {
        for (engine::SlotId s = 0; s < active_.size(); ++s) {
          if (active_[s].job == job) {
            on_done(s, finish, stats);
            return;
          }
        }
      });
  if (!slot) {
    fail(*job, slot.error().message);
    return;
  }
  if (!slot->has_value()) {
    fail(*job, "no free slot although one was counted");  // cannot happen, admitted under running()
    return;
  }
  active_[**slot] = Active{.job = std::move(job), .matcher = std::move(matcher)};
  running_.store(batch_.running());
  active_[**slot].job->batch_size_ = running_.load();
}

void Worker::on_text(engine::SlotId slot, std::string_view piece) {
  Active& a = active_[slot];
  std::string release = a.matcher->feed(piece);
  if (!release.empty()) {
    a.job->push(std::move(release));
  }
  if (a.matcher->hit()) {
    a.job->abort();  // a stop string completed, the sequence ends at its next step
  }
}

void Worker::on_done(engine::SlotId slot, engine::Finish finish, const engine::Stats& stats) {
  Active& a = active_[slot];
  std::string tail = a.matcher->flush();
  if (!tail.empty()) {
    a.job->push(std::move(tail));
  }
  std::string reason = "stop";
  if (finish == engine::Finish::Length) {
    reason = "length";
  }
  a.job->finish(
      reason,
      Usage{.prompt_tokens = stats.prompt_tokens, .completion_tokens = stats.generated_tokens},
      std::nullopt);
  a.job.reset();
  a.matcher.reset();
  served_.fetch_add(1);
}

void Worker::run() {
  for (;;) {
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || !queue_.empty() || batch_.busy(); });
      if (stopping_) {
        for (auto& job : queue_) {
          job->finish("stop", {},
                      ApiError{.status = 503,
                               .type = "server_error",
                               .code = "shutting_down",
                               .message = "the server is shutting down",
                               .param = std::nullopt});
        }
        queue_.clear();
        for (auto& a : active_) {
          if (a.job) {
            fail(*a.job, "the server is shutting down");
            a.job.reset();
            a.matcher.reset();
          }
        }
        return;
      }
      while (!queue_.empty() && batch_.running() < batch_.slots()) {
        std::shared_ptr<Job> job = std::move(queue_.front());
        queue_.pop_front();
        lock.unlock();
        admit(std::move(job));
        lock.lock();
      }
    }
    if (!batch_.busy()) {
      continue;
    }
    const auto finished = batch_.step();
    batched_steps_.store(batch_.batched_steps());
    running_.store(batch_.running());
    if (!finished) {
      // The pass itself failed: every running job gets the error and its slot is freed.
      for (auto& a : active_) {
        if (a.job) {
          fail(*a.job, finished.error().message);
          a.job.reset();
          a.matcher.reset();
        }
      }
    }
  }
}

}  // namespace axon::server
