#pragma once

// Perplexity over a token stream, scored the way llama.cpp's perplexity tool does: the stream
// is cut into chunks, each chunk is one prefill, and the second half of every chunk is scored
// with the first half as context.

#include "core/expected.h"

#include "model/llama.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace axon::engine {

struct PerplexityOptions {
  std::size_t chunk_size = 512;
  std::size_t max_chunks = 0;  // 0 scores every complete chunk
};

struct PerplexityResult {
  double nll_sum = 0.0;
  std::size_t scored_tokens = 0;
  std::size_t chunks = 0;
  std::vector<double> running;  // perplexity after each chunk

  [[nodiscard]] double perplexity() const {
    return scored_tokens == 0 ? 0.0 : std::exp(nll_sum / static_cast<double>(scored_tokens));
  }
};

// The model must have been loaded with max_ctx >= chunk_size. Progress is called after each
// chunk with its index and the running perplexity.
Expected<PerplexityResult>
perplexity(model::LlamaModel& model, std::span<const std::int32_t> tokens,
           const PerplexityOptions& opts,
           const std::function<void(std::size_t, double)>& progress = {});

}  // namespace axon::engine
