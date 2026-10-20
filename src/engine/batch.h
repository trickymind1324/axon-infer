#pragma once

// Several generations at once from one model. Each slot is a Sequence with its own cache. A
// step prefills one chunk of the oldest admitted prompt that still has prompt left, then
// advances every sequence in decode by one token through a single batched forward pass, so
// the weights are read once for the whole batch. Each row of that pass equals what the
// sequence would compute alone, so the text per sequence is the single-engine text.

#include "core/expected.h"

#include "engine/options.h"
#include "engine/sequence.h"
#include "model/llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace axon::engine {

using SlotId = std::size_t;

class BatchEngine {
public:
  // The model must have been loaded with max_batch at least `slots`.
  BatchEngine(model::LlamaModel& model, std::size_t slots);

  using TextSink = std::function<void(std::string_view)>;
  using DoneSink = std::function<void(Finish, const Stats&)>;
  // Admits a prompt into the free slot whose cache shares the longest prefix with it, or
  // returns nullopt when every slot is busy. Fails like Engine::generate when the prompt
  // cannot be processed. `done` is called from step() when the generation ends.
  Expected<std::optional<SlotId>> admit(std::span<const std::int32_t> prompt,
                                        const GenerateOptions& opts, TextSink on_text,
                                        DoneSink done);
  // One scheduler turn. Returns how many sequences finished during it.
  Expected<std::size_t> step();

  [[nodiscard]] std::size_t slots() const noexcept { return sequences_.size(); }
  [[nodiscard]] std::size_t running() const;  // slots in prefill or decode
  [[nodiscard]] bool busy() const { return running() > 0; }
  [[nodiscard]] const Sequence& sequence(SlotId id) const { return *sequences_[id]; }
  // Decode steps that advanced more than one sequence, for tests and the stats line.
  [[nodiscard]] std::uint64_t batched_steps() const noexcept { return batched_steps_; }
  [[nodiscard]] std::uint64_t decode_steps() const noexcept { return decode_steps_; }

private:
  struct Slot {
    DoneSink done;
    std::uint64_t admitted_at = 0;  // admission order, for the prefill choice
  };
  void finish(SlotId id);

  model::LlamaModel& model_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
  std::vector<Slot> slots_;  // one per sequence
  std::uint64_t admissions_ = 0;
  std::uint64_t batched_steps_ = 0;
  std::uint64_t decode_steps_ = 0;
  // Scratch for a step, kept across calls so a step allocates nothing.
  std::vector<model::KVCache*> caches_;
  std::vector<std::int32_t> tokens_;
  std::vector<SlotId> decoding_;
};

}  // namespace axon::engine
