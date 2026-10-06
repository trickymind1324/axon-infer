#include "kernels/kernels.h"

#include "core/cpu_info.h"

#include "kernels/ref/ref.h"
#if defined(AXON_HAVE_NEON)
#include "kernels/neon/neon.h"
#endif

#include <fmt/format.h>

namespace axon::kernels {

namespace {

constexpr KernelTable kScalarTable{
    .matmul_f32 = ref::matmul_f32,
    .matmul_f16 = ref::matmul_f16,
    .matmul_bf16 = ref::matmul_bf16,
    .matmul_f32_rows = ref::matmul_f32_rows,
    .matmul_f16_rows = ref::matmul_f16_rows,
    .matmul_bf16_rows = ref::matmul_bf16_rows,
    .rmsnorm = ref::rmsnorm,
    .rope = ref::rope,
    .softmax = ref::softmax,
    .silu_mul = ref::silu_mul,
    .embedding_row = ref::embedding_row,
    .attention = ref::attention,
    .attention_kv = ref::attention_kv,
    .attention_kv_heads = ref::attention_kv_heads,
    .matmul_q8_0 = ref::matmul_q8_0,
    .matmul_q4_0 = ref::matmul_q4_0,
    .matmul_q8_0_q8 = ref::matmul_q8_0_q8,
    .matmul_q4_0_q8 = ref::matmul_q4_0_q8,
    .matmul_q8_0_rows = ref::matmul_q8_0_rows,
    .matmul_q4_0_rows = ref::matmul_q4_0_rows,
    .matmul_q8_0_q8_rows = ref::matmul_q8_0_q8_rows,
    .matmul_q4_0_q8_rows = ref::matmul_q4_0_q8_rows,
    .quantize_row_q8_0 = ref::quantize_row_q8_0,
    .matmul_q4_1 = ref::matmul_q4_1,
    .matmul_q4_1_q8 = ref::matmul_q4_1_q8,
    .matmul_q4_1_rows = ref::matmul_q4_1_rows,
    .matmul_q4_1_q8_rows = ref::matmul_q4_1_q8_rows,
    .matmul_q6_k = ref::matmul_q6_k,
    .matmul_q6_k_rows = ref::matmul_q6_k_rows,
};

#if defined(AXON_HAVE_NEON)
const KernelTable& neon_table() {
  static const KernelTable kTable = [] {
    KernelTable t = kScalarTable;
    neon::install(t);
    return t;
  }();
  return kTable;
}
#endif

bool neon_usable() {
#if defined(AXON_HAVE_NEON)
  const CpuInfo& cpu = cpu_info();
  return cpu.neon && cpu.neon_dotprod;
#else
  return false;
#endif
}

}  // namespace

const KernelTable& scalar_kernels() {
  return kScalarTable;
}

const KernelTable& kernels() {
  return kScalarTable;
}

Expected<const KernelTable*> select(KernelSet set) {
  switch (set) {
  case KernelSet::Scalar:
    return &kScalarTable;
  case KernelSet::Neon:
#if defined(AXON_HAVE_NEON)
    if (!neon_usable()) {
      return make_error(ErrorCode::Unsupported,
                        "kernels: this CPU lacks the NEON dot product extension (FEAT_DotProd)");
    }
    return &neon_table();
#else
    return make_error(ErrorCode::Unsupported,
                      "kernels: NEON kernels are not built on this platform");
#endif
  case KernelSet::Auto:
#if defined(AXON_HAVE_NEON)
    if (neon_usable()) {
      return &neon_table();
    }
#endif
    return &kScalarTable;
  }
  return &kScalarTable;
}

std::string_view name(const KernelTable& table) {
#if defined(AXON_HAVE_NEON)
  if (&table == &neon_table()) {
    return "neon";
  }
#endif
  return &table == &kScalarTable ? "scalar" : "custom";
}

bool parse_kernel_set(std::string_view text, KernelSet& out) {
  if (text == "auto") {
    out = KernelSet::Auto;
  } else if (text == "scalar") {
    out = KernelSet::Scalar;
  } else if (text == "neon") {
    out = KernelSet::Neon;
  } else {
    return false;
  }
  return true;
}

}  // namespace axon::kernels
