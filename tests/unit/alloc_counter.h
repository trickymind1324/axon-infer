#pragma once

#include <cstddef>

// Global operator new/delete are replaced in alloc_counter.cpp so tests can assert that a
// code path performs no heap allocation.
namespace axon::test {

std::size_t allocation_count();

}  // namespace axon::test
