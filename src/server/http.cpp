#include "server/http.h"

#include <fmt/format.h>

#include <httplib.h>

#include <chrono>
#include <exception>
#include <optional>
#include <utility>

namespace axon::server {

namespace {

using nlohmann::json;

constexpr const char* kJson = "application/json";
constexpr std::chrono::milliseconds kPollInterval{100};

void respond_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), kJson);
}

void respond_error(httplib::Response& res, const ApiError& error) {
  if (error.status == 503) {
    res.set_header("Retry-After", "1");
  }
  respond_json(res, error.status, error_json(error));
}

// Streams one job as server-sent events. Each call of the provider delivers one event, so
// httplib paces the writes, and a client that has gone cancels the job at the next poll.
struct Stream {
  std::shared_ptr<Job> job;
  std::string id;
  std::int64_t created;
  bool sent_role = false;
  bool finished = false;

  bool write(httplib::DataSink& sink, const std::string& text) {
    if (!sink.write(text.data(), text.size())) {
      job->cancel();
      return false;
    }
    return true;
  }

  bool step(httplib::DataSink& sink) {
    if (finished) {
      sink.done();
      return true;
    }
    const CompletionRequest& req = job->request();
    if (req.chat && !sent_role) {
      sent_role = true;
      return write(sink, sse_event(chunk_json(req, id, created, "assistant", "", std::nullopt,
                                              std::nullopt)));
    }
    for (;;) {
      std::string piece;
      switch (job->next(piece, kPollInterval)) {
      case Job::Poll::Piece:
        return write(sink, sse_event(chunk_json(req, id, created, std::nullopt, piece, std::nullopt,
                                                std::nullopt)));
      case Job::Poll::Done: {
        finished = true;
        std::string out;
        if (const auto error = job->error()) {
          out = sse_event(error_json(*error));
        } else {
          const std::optional<Usage> usage =
              req.include_usage ? std::optional<Usage>(job->usage()) : std::nullopt;
          out = sse_event(
              chunk_json(req, id, created, std::nullopt, "", job->finish_reason(), usage));
        }
        out += kSseDone;
        if (!write(sink, out)) {
          return false;
        }
        sink.done();
        return true;
      }
      case Job::Poll::Timeout:
        if (!sink.is_writable()) {
          job->cancel();
          return false;
        }
        break;
      }
    }
  }
};

}  // namespace

struct HttpServer::Impl {
  httplib::Server server;
};

HttpServer::HttpServer(model::LlamaModel& model, ServerOptions options)
    : options_(std::move(options)),
      template_(chat::ChatTemplate::from_tokenizer(model.tokenizer())), engine_(model),
      impl_(std::make_unique<Impl>()) {
  if (options_.served_name.empty()) {
    options_.served_name = std::string(model.file().model_name());
  }
  context_ = Context{.tokenizer = &model.tokenizer(),
                     .chat_template = &template_,
                     .served_name = options_.served_name,
                     .default_max_tokens = options_.default_max_tokens,
                     .max_ctx = model.max_ctx()};
  worker_ = std::make_unique<Worker>(engine_, options_.max_queue);
  routes();
}

HttpServer::~HttpServer() {
  stop();
}

void HttpServer::routes() {
  httplib::Server& svr = impl_->server;
  svr.set_payload_max_length(static_cast<std::size_t>(8) * 1024 * 1024);
  svr.set_read_timeout(std::chrono::seconds(30));
  svr.set_write_timeout(std::chrono::seconds(30));
  svr.set_keep_alive_timeout(std::chrono::seconds(10));

  if (!options_.api_key.empty()) {
    const std::string expected = "Bearer " + options_.api_key;
    svr.set_pre_routing_handler([expected](const httplib::Request& req, httplib::Response& res) {
      // Runs before routing, where an exception would have no handler, so it keeps its own.
      try {
        if (req.path.starts_with("/v1") && req.get_header_value("Authorization") != expected) {
          res.set_header("WWW-Authenticate", "Bearer");
          respond_error(res, ApiError{.status = 401,
                                      .type = "authentication_error",
                                      .code = "invalid_api_key",
                                      .message = "a valid bearer token is required",
                                      .param = std::nullopt});
          return httplib::Server::HandlerResponse::Handled;
        }
      } catch (const std::exception&) {
        respond_error(res, ApiError{.status = 500,
                                    .type = "server_error",
                                    .code = "internal_error",
                                    .message = "authentication check failed",
                                    .param = std::nullopt});
        return httplib::Server::HandlerResponse::Handled;
      }
      return httplib::Server::HandlerResponse::Unhandled;
    });
  }

  svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (!res.body.empty()) {
      return;  // a handler already wrote the error
    }
    respond_error(
        res, ApiError{.status = res.status,
                      .type = res.status == 404 ? "not_found_error" : "server_error",
                      .code = res.status == 404 ? "not_found" : "",
                      .message = res.status == 404
                                     ? fmt::format("no route for {} {}", req.method, req.path)
                                     : fmt::format("request failed with status {}", res.status),
                      .param = std::nullopt});
  });
  svr.set_exception_handler(
      [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown error";
        try {
          std::rethrow_exception(std::move(ep));
        } catch (const std::exception& e) {
          what = e.what();
        } catch (...) {
          what = "an exception that is not a std::exception";
        }
        respond_error(res, ApiError{.status = 500,
                                    .type = "server_error",
                                    .code = "internal_error",
                                    .message = what,
                                    .param = std::nullopt});
      });

  svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    const auto uptime = static_cast<double>(now_unix() - started_unix_);
    respond_json(
        res, 200,
        health_json(options_.served_name, uptime, worker_->served(), worker_->queue_depth()));
  });
  svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    respond_json(res, 200, models_json(options_.served_name, started_unix_));
  });

  const auto completion = [this](bool chat) {
    return [this, chat](const httplib::Request& req, httplib::Response& res) {
      auto body = parse_body(req.body);
      if (std::holds_alternative<ApiError>(body)) {
        respond_error(res, std::get<ApiError>(body));
        return;
      }
      auto parsed = parse_request(std::get<json>(body), context_, chat);
      if (std::holds_alternative<ApiError>(parsed)) {
        respond_error(res, std::get<ApiError>(parsed));
        return;
      }
      auto job = worker_->submit(std::get<CompletionRequest>(std::move(parsed)));
      if (!job) {
        respond_error(res, ApiError{.status = 503,
                                    .type = "server_error",
                                    .code = "queue_full",
                                    .message = fmt::format("{} requests are already waiting",
                                                           options_.max_queue),
                                    .param = std::nullopt});
        return;
      }
      const std::string id = completion_id(chat);
      const std::int64_t created = now_unix();
      if (job->request().stream) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        auto stream = std::make_shared<Stream>(Stream{.job = job, .id = id, .created = created});
        res.set_chunked_content_provider(
            "text/event-stream",
            [stream](std::size_t, httplib::DataSink& sink) { return stream->step(sink); },
            [stream](bool success) {
              if (!success) {
                stream->job->cancel();
              }
            });
        return;
      }
      std::string text;
      for (;;) {
        std::string piece;
        const auto poll = job->next(piece, kPollInterval);
        if (poll == Job::Poll::Piece) {
          text += piece;
        } else if (poll == Job::Poll::Done) {
          break;
        }
      }
      if (const auto error = job->error()) {
        respond_error(res, *error);
        return;
      }
      res.set_header("X-Axon-Queue-Ms", fmt::format("{:.0f}", 1e3 * job->queue_wait_s()));
      respond_json(
          res, 200,
          completion_json(job->request(), id, created, text, job->finish_reason(), job->usage()));
    };
  };
  svr.Post("/v1/chat/completions", completion(true));
  svr.Post("/v1/completions", completion(false));
}

bool HttpServer::start() {
  httplib::Server& svr = impl_->server;
  if (options_.port == 0) {
    port_ = svr.bind_to_any_port(options_.host);
    if (port_ <= 0) {
      return false;
    }
  } else {
    if (!svr.bind_to_port(options_.host, options_.port)) {
      return false;
    }
    port_ = options_.port;
  }
  started_unix_ = now_unix();
  thread_ = std::thread([this] { impl_->server.listen_after_bind(); });
  svr.wait_until_ready();
  return true;
}

void HttpServer::stop() {
  if (impl_ && impl_->server.is_running()) {
    impl_->server.stop();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
  if (worker_) {
    worker_->stop();
  }
}

}  // namespace axon::server
