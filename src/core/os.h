#pragma once

#include "core/expected.h"

#include <cstddef>
#include <filesystem>

// Thin platform layer. Everything OS-specific for memory lives here so that the arena and
// the GGUF reader never mention the platform.
namespace axon::os {

std::size_t page_size();

// Anonymous, private, read-write, page-aligned mapping. Pages are committed on first touch.
// Returns nullptr on failure.
void* map_anonymous(std::size_t bytes);
void unmap(void* address, std::size_t bytes) noexcept;

// Read-only, private mapping of a whole file. Empty files are rejected.
struct FileMapping {
  const std::byte* data = nullptr;
  std::size_t size = 0;
};
Expected<FileMapping> map_file(const std::filesystem::path& path);
void unmap_file(FileMapping& mapping) noexcept;

// Hint that the mapping will be read front to back once.
void advise_sequential(const FileMapping& mapping) noexcept;

}  // namespace axon::os
