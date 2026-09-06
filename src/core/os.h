#pragma once

#include <cstddef>

// Thin platform layer. Everything OS-specific for memory lives here so that the arena and
// (later) the GGUF mapper never mention the platform.
namespace axon::os {

std::size_t page_size();

// Anonymous, private, read-write, page-aligned mapping. Pages are committed on first touch.
// Returns nullptr on failure.
void* map_anonymous(std::size_t bytes);
void unmap(void* address, std::size_t bytes) noexcept;

}  // namespace axon::os
