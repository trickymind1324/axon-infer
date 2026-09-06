#pragma once

#include <string>

namespace axon {

// Static description of the host CPU, computed once. Used for thread-pool defaults,
// diagnostics, and (from the SIMD phase) runtime kernel dispatch.
struct CpuInfo {
  int logical_cores = 1;
  int physical_cores = 1;
  int cache_line_bytes = 64;

  // arm64
  bool neon = false;
  bool neon_dotprod = false;
  bool neon_fp16 = false;

  // x86-64
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx512f = false;
  bool avx512_vnni = false;

  std::string arch;   // "arm64" or "x86_64"
  std::string brand;  // marketing name when available

  // One-line summary for --version and logs.
  [[nodiscard]] std::string summary() const;
};

const CpuInfo& cpu_info();

}  // namespace axon
