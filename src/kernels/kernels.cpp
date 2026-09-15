#include "kernels/kernels.h"

#include "kernels/ref/ref.h"

namespace axon::kernels {

namespace {

constexpr KernelTable kScalarTable{
    .rmsnorm = ref::rmsnorm,
    .softmax = ref::softmax,
    .silu_mul = ref::silu_mul,
};

}  // namespace

const KernelTable& scalar_kernels() {
  return kScalarTable;
}

const KernelTable& kernels() {
  return kScalarTable;
}

}  // namespace axon::kernels
