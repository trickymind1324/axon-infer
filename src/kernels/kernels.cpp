#include "kernels/kernels.h"

#include "kernels/ref/ref.h"

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
};

}  // namespace

const KernelTable& scalar_kernels() {
  return kScalarTable;
}

const KernelTable& kernels() {
  return kScalarTable;
}

}  // namespace axon::kernels
