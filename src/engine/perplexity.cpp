#include "engine/perplexity.h"

#include <fmt/format.h>

#include <algorithm>

namespace axon::engine {

namespace {

// -log softmax(logits)[target], in double.
double nll_of(std::span<const float> row, std::int32_t target) {
  auto max = static_cast<double>(row[0]);
  for (const float v : row) {
    max = std::max(max, static_cast<double>(v));
  }
  double sum = 0.0;
  for (const float v : row) {
    sum += std::exp(static_cast<double>(v) - max);
  }
  return max + std::log(sum) - static_cast<double>(row[static_cast<std::size_t>(target)]);
}

}  // namespace

Expected<PerplexityResult> perplexity(model::LlamaModel& model,
                                      std::span<const std::int32_t> tokens,
                                      const PerplexityOptions& opts,
                                      const std::function<void(std::size_t, double)>& progress) {
  const std::size_t chunk = opts.chunk_size;
  if (chunk < 4 || chunk % 2 != 0) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("perplexity: chunk size must be even and at least 4, found {}", chunk));
  }
  if (model.max_ctx() < chunk) {
    return make_error(ErrorCode::InvalidArgument,
                      fmt::format("perplexity: the model was loaded with max_ctx {}, smaller "
                                  "than the chunk size {}",
                                  model.max_ctx(), chunk));
  }
  std::size_t n_chunks = tokens.size() / chunk;
  if (opts.max_chunks != 0) {
    n_chunks = std::min(n_chunks, opts.max_chunks);
  }
  if (n_chunks == 0) {
    return make_error(
        ErrorCode::InvalidArgument,
        fmt::format("perplexity: {} tokens make no complete chunk of {}", tokens.size(), chunk));
  }
  const std::size_t n_vocab = model.config().n_vocab;
  const std::size_t first_scored = chunk / 2;  // position whose prediction is scored first
  PerplexityResult result;
  auto cache = model.new_cache();
  for (std::size_t c = 0; c < n_chunks; ++c) {
    const auto window = tokens.subspan(c * chunk, chunk);
    cache.clear();
    // Prefill in pieces of max_seq. A piece's logits at position p predict window[p + 1].
    for (std::size_t offset = 0; offset < chunk; offset += model.max_seq()) {
      const std::size_t n = std::min(model.max_seq(), chunk - offset);
      auto logits = model.forward_prefill(cache, window.subspan(offset, n));
      if (!logits) {
        return logits.error();
      }
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t p = offset + i;
        if (p + 1 < first_scored || p + 1 >= chunk) {
          continue;
        }
        result.nll_sum += nll_of(logits->subspan(i * n_vocab, n_vocab), window[p + 1]);
        ++result.scored_tokens;
      }
    }
    ++result.chunks;
    result.running.push_back(result.perplexity());
    if (progress) {
      progress(c, result.running.back());
    }
  }
  return result;
}

}  // namespace axon::engine
