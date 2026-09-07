#include "core/gguf.h"

#include <cstddef>
#include <cstdint>
#include <span>

// libFuzzer entry point: the parser must return a value or an error for any input.
// Build with the `fuzz` preset (Linux clang):
//   ./build/fuzz/tests/fuzz/axon_fuzz_gguf tests/reference/fixtures/gguf -max_total_time=60
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
  const auto file = axon::GGUFFile::parse(bytes);
  if (file) {
    volatile std::size_t sink = 0;
    for (const auto& [key, value] : file->metadata_entries()) {
      sink = sink + key.size() + value.array_size();
    }
    for (const auto& t : file->tensors()) {
      sink = sink + t.name.size() + (t.nbytes > 0 ? static_cast<std::size_t>(t.data[0]) : 0);
    }
  }
  return 0;
}
