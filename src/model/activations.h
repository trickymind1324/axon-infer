#pragma once

// The forward pass's working buffers. One function lays them out so the arena size computed at
// load and the allocations made per forward can never disagree: it runs once against a
// SizeCounter and then against the real Arena.

#include "core/arena.h"

#include "model/config.h"

#include <cstddef>
#include <span>

namespace axon::model {

struct Activations {
  std::span<float> x;       // [seq][n_embd] residual stream
  std::span<float> h;       // [seq][n_embd] normed input, then projection outputs
  std::span<float> q;       // [seq][n_head * head_dim]
  std::span<float> k;       // [seq][n_head_kv * head_dim]
  std::span<float> v;       // [seq][n_head_kv * head_dim]
  std::span<float> attn;    // [seq][n_head * head_dim]
  std::span<float> gate;    // [seq][n_ff]
  std::span<float> up;      // [seq][n_ff]
  std::span<float> scores;  // [seq][seq] attention scratch
  std::span<float> logits;  // [seq][n_vocab]
};

template <class Alloc>
Activations allocate_activations(Alloc& alloc, const LlamaConfig& cfg, std::size_t seq) {
  Activations a;
  a.x = alloc.template alloc_array<float>(seq * cfg.n_embd);
  a.h = alloc.template alloc_array<float>(seq * cfg.n_embd);
  a.q = alloc.template alloc_array<float>(seq * cfg.n_embd_q());
  a.k = alloc.template alloc_array<float>(seq * cfg.n_embd_kv());
  a.v = alloc.template alloc_array<float>(seq * cfg.n_embd_kv());
  a.attn = alloc.template alloc_array<float>(seq * cfg.n_embd_q());
  a.gate = alloc.template alloc_array<float>(seq * cfg.n_ff);
  a.up = alloc.template alloc_array<float>(seq * cfg.n_ff);
  a.scores = alloc.template alloc_array<float>(seq * seq);
  a.logits = alloc.template alloc_array<float>(seq * cfg.n_vocab);
  return a;
}

// Counts what the Arena would hand out, rounding each block like Arena::alloc does.
struct SizeCounter {
  std::size_t total = 0;

  template <class T> std::span<T> alloc_array(std::size_t count) {
    const std::size_t bytes = count * sizeof(T);
    total += (bytes + Arena::kDefaultAlign - 1) / Arena::kDefaultAlign * Arena::kDefaultAlign;
    return {};
  }
};

}  // namespace axon::model
