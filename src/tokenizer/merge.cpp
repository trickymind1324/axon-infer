#include "tokenizer/merge.h"

#include <queue>

namespace axon {

MergeRanks::MergeRanks(const std::vector<std::string_view>& merges) {
  ranks_.reserve(merges.size() * 2);
  for (std::size_t i = 0; i < merges.size(); ++i) {
    ranks_.emplace(std::string(merges[i]), static_cast<std::int32_t>(i));
  }
}

std::int32_t MergeRanks::rank(std::string_view left, std::string_view right) const {
  key_.assign(left);
  key_.push_back(' ');
  key_.append(right);
  const auto it = ranks_.find(key_);
  return it == ranks_.end() ? -1 : it->second;
}

namespace {

struct Candidate {
  float priority;  // lower merges first
  int left;
  std::size_t left_end;
  std::size_t right_end;
  bool operator<(const Candidate& other) const noexcept {
    return priority != other.priority ? priority > other.priority : left > other.left;
  }
};

template <class Priority>
void merge_loop(std::string_view text, std::vector<MergeSymbol>& symbols, Priority priority) {
  if (symbols.empty()) {
    return;
  }
  auto piece = [&](const MergeSymbol& s) { return text.substr(s.begin, s.end - s.begin); };
  std::priority_queue<Candidate> queue;
  auto consider = [&](int left) {
    if (left < 0) {
      return;
    }
    const MergeSymbol& l = symbols[static_cast<std::size_t>(left)];
    if (l.next < 0) {
      return;
    }
    const MergeSymbol& r = symbols[static_cast<std::size_t>(l.next)];
    float p = 0.0F;
    if (priority(piece(l), piece(r), p)) {
      queue.push({p, left, l.end, r.end});
    }
  };
  for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
    consider(i);
  }
  while (!queue.empty()) {
    const Candidate c = queue.top();
    queue.pop();
    MergeSymbol& l = symbols[static_cast<std::size_t>(c.left)];
    if (!l.alive() || l.end != c.left_end || l.next < 0) {
      continue;  // stale entry
    }
    MergeSymbol& r = symbols[static_cast<std::size_t>(l.next)];
    if (!r.alive() || r.end != c.right_end || r.begin != l.end) {
      continue;
    }
    l.end = r.end;
    l.next = r.next;
    r.begin = r.end = 0;
    if (l.next >= 0) {
      symbols[static_cast<std::size_t>(l.next)].prev = c.left;
    }
    consider(l.prev);
    consider(c.left);
  }
}

}  // namespace

void merge_by_rank(std::string_view text, std::vector<MergeSymbol>& symbols,
                   const MergeRanks& ranks) {
  merge_loop(text, symbols, [&](std::string_view l, std::string_view r, float& p) {
    const std::int32_t rank = ranks.rank(l, r);
    if (rank < 0) {
      return false;
    }
    p = static_cast<float>(rank);
    return true;
  });
}

void merge_by_score(std::string_view text, std::vector<MergeSymbol>& symbols,
                    FunctionRef<bool(std::string_view, float&)> score) {
  std::string merged;
  merge_loop(text, symbols, [&](std::string_view l, std::string_view r, float& p) {
    merged.assign(l);
    merged.append(r);
    float s = 0.0F;
    if (!score(merged, s)) {
      return false;
    }
    p = -s;  // highest score first
    return true;
  });
}

}  // namespace axon
