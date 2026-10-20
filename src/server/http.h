#pragma once

// The routes over a Worker: health, the model list, and the two completion endpoints with
// and without server-sent events. The server runs on its own thread so a test can start it
// on a free port in process, talk to it, and stop it.

#include "chat/template.h"
#include "engine/batch.h"
#include "model/llama.h"
#include "server/api.h"
#include "server/worker.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace axon::server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;  // 0 picks a free port, read back with port()
  std::size_t max_queue = 16;
  std::size_t max_batch = 4;  // sequences decoded together, capped by the model's max_batch
  std::string api_key;        // empty means no authentication
  std::string served_name;    // empty means the model file's stem
  int default_max_tokens = 512;
};

class HttpServer {
public:
  HttpServer(model::LlamaModel& model, ServerOptions options);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds and starts serving on a background thread. False when the port cannot be bound.
  bool start();
  // Stops accepting, ends the job in flight, joins the thread.
  void stop();
  [[nodiscard]] int port() const noexcept { return port_; }
  [[nodiscard]] const std::string& served_name() const noexcept { return options_.served_name; }
  [[nodiscard]] std::uint64_t requests() const noexcept { return worker_->served(); }

private:
  struct Impl;  // keeps httplib out of this header
  void routes();

  ServerOptions options_;
  std::string bearer_;  // "Bearer <key>" when a key is set
  chat::ChatTemplate template_;
  Context context_;
  engine::BatchEngine batch_;
  std::unique_ptr<Worker> worker_;
  std::unique_ptr<Impl> impl_;
  std::thread thread_;
  int port_ = 0;
  std::int64_t started_unix_ = 0;
};

}  // namespace axon::server
