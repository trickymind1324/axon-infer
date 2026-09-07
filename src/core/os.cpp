#include "core/os.h"

#include <fmt/format.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

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

Expected<FileMapping> map_file(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return make_error(ErrorCode::Io,
                      fmt::format("cannot open '{}': {}", path.string(), std::strerror(errno)));
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    const int saved = errno;
    ::close(fd);
    return make_error(ErrorCode::Io,
                      fmt::format("cannot stat '{}': {}", path.string(), std::strerror(saved)));
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    return make_error(ErrorCode::Io, fmt::format("'{}': expected a regular file", path.string()));
  }
  if (st.st_size <= 0) {
    ::close(fd);
    return make_error(ErrorCode::Io, fmt::format("'{}': file is empty", path.string()));
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  void* address = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int saved = errno;
  ::close(fd);
  if (address == MAP_FAILED) {
    return make_error(ErrorCode::Io, fmt::format("cannot map '{}' ({} bytes): {}", path.string(),
                                                 size, std::strerror(saved)));
  }
  return FileMapping{.data = static_cast<const std::byte*>(address), .size = size};
}

void unmap_file(FileMapping& mapping) noexcept {
  if (mapping.data != nullptr) {
    ::munmap(const_cast<std::byte*>(mapping.data), mapping.size);
    mapping.data = nullptr;
    mapping.size = 0;
  }
}

void advise_sequential(const FileMapping& mapping) noexcept {
  if (mapping.data != nullptr) {
    ::madvise(const_cast<std::byte*>(mapping.data), mapping.size, MADV_SEQUENTIAL);
  }
}

}  // namespace axon::os
