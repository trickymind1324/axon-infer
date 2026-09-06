#include "core/os.h"

#include <sys/mman.h>
#include <unistd.h>

namespace axon::os {

std::size_t page_size() {
  static const auto kPage = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  return kPage;
}

void* map_anonymous(std::size_t bytes) {
  void* address =
      ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return address == MAP_FAILED ? nullptr : address;
}

void unmap(void* address, std::size_t bytes) noexcept {
  ::munmap(address, bytes);
}

}  // namespace axon::os
