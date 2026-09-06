#include "core/cpu_info.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>

#include <fstream>
#include <set>
#include <sstream>
#endif

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace axon {

namespace {

#if defined(__APPLE__)
bool sysctl_int(const char* name, long long& out) {
  long long value = 0;
  std::size_t size = sizeof(value);
  if (::sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
    return false;
  }
  out = value;
  return true;
}

std::string sysctl_string(const char* name) {
  std::size_t size = 0;
  if (::sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return {};
  }
  std::string value(size, '\0');
  if (::sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
    return {};
  }
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}
#endif

#if defined(__linux__)
std::string cpuinfo_field(const char* key) {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  const std::string prefix(key);
  while (std::getline(in, line)) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) {
        auto value = line.substr(colon + 1);
        const auto first = value.find_first_not_of(" \t");
        return first == std::string::npos ? std::string{} : value.substr(first);
      }
    }
  }
  return {};
}

int linux_physical_cores() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  std::set<std::pair<int, int>> cores;
  int physical_id = 0;
  int core_id = -1;
  while (std::getline(in, line)) {
    if (line.rfind("physical id", 0) == 0) {
      physical_id = std::stoi(line.substr(line.find(':') + 1));
    } else if (line.rfind("core id", 0) == 0) {
      core_id = std::stoi(line.substr(line.find(':') + 1));
    } else if (line.empty() && core_id >= 0) {
      cores.emplace(physical_id, core_id);
      core_id = -1;
    }
  }
  if (core_id >= 0) {
    cores.emplace(physical_id, core_id);
  }
  return cores.empty() ? 0 : static_cast<int>(cores.size());
}
#endif

CpuInfo detect() {
  CpuInfo info;
  info.logical_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  info.physical_cores = info.logical_cores;

#if defined(__aarch64__)
  info.arch = "arm64";
  info.neon = true;  // baseline for AArch64
#elif defined(__x86_64__)
  info.arch = "x86_64";
#else
  info.arch = "unknown";
#endif

#if defined(__APPLE__)
  long long value = 0;
  if (sysctl_int("hw.physicalcpu", value) && value > 0) {
    info.physical_cores = static_cast<int>(value);
  }
  if (sysctl_int("hw.cachelinesize", value) && value > 0) {
    info.cache_line_bytes = static_cast<int>(value);
  }
  info.brand = sysctl_string("machdep.cpu.brand_string");
#if defined(__aarch64__)
  if (sysctl_int("hw.optional.arm.FEAT_DotProd", value)) {
    info.neon_dotprod = value != 0;
  }
  if (sysctl_int("hw.optional.arm.FEAT_FP16", value)) {
    info.neon_fp16 = value != 0;
  }
#endif
#endif

#if defined(__linux__)
  if (const int physical = linux_physical_cores(); physical > 0) {
    info.physical_cores = physical;
  }
  if (const long line = ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE); line > 0) {
    info.cache_line_bytes = static_cast<int>(line);
  }
  info.brand = cpuinfo_field("model name");
#if defined(__aarch64__)
  const unsigned long hwcap = ::getauxval(AT_HWCAP);
  info.neon_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
  info.neon_fp16 = (hwcap & HWCAP_ASIMDHP) != 0;
#endif
#endif

#if defined(__x86_64__)
  unsigned eax = 0;
  unsigned ebx = 0;
  unsigned ecx = 0;
  unsigned edx = 0;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
    info.fma = (ecx & (1U << 12)) != 0;
    info.f16c = (ecx & (1U << 29)) != 0;
  }
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
    info.avx2 = (ebx & (1U << 5)) != 0;
    info.avx512f = (ebx & (1U << 16)) != 0;
    info.avx512_vnni = (ecx & (1U << 11)) != 0;
  }
  if (info.brand.empty()) {
    char brand[49] = {};
    unsigned regs[12] = {};
    for (unsigned i = 0; i < 3; ++i) {
      __get_cpuid(0x80000002U + i, &regs[i * 4], &regs[i * 4 + 1], &regs[i * 4 + 2],
                  &regs[i * 4 + 3]);
    }
    std::memcpy(brand, regs, sizeof(regs));
    info.brand = brand;
  }
#endif

  if (info.brand.empty()) {
    info.brand = "unknown";
  }
  return info;
}

}  // namespace

std::string CpuInfo::summary() const {
  std::vector<std::string> features;
  if (neon) {
    features.emplace_back("neon");
  }
  if (neon_dotprod) {
    features.emplace_back("dotprod");
  }
  if (neon_fp16) {
    features.emplace_back("fp16");
  }
  if (avx2) {
    features.emplace_back("avx2");
  }
  if (fma) {
    features.emplace_back("fma");
  }
  if (f16c) {
    features.emplace_back("f16c");
  }
  if (avx512f) {
    features.emplace_back("avx512f");
  }
  if (avx512_vnni) {
    features.emplace_back("avx512_vnni");
  }
  std::string feature_list;
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (i != 0) {
      feature_list += ',';
    }
    feature_list += features[i];
  }
  if (feature_list.empty()) {
    feature_list = "none";
  }
  return fmt::format("{} ({}) cores={}/{} cacheline={} features={}", brand, arch, physical_cores,
                     logical_cores, cache_line_bytes, feature_list);
}

const CpuInfo& cpu_info() {
  static const CpuInfo kInfo = detect();
  return kInfo;
}

}  // namespace axon
