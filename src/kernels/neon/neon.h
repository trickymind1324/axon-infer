#pragma once

// NEON implementations, compiled on arm64 only. Each file overrides the table entries it
// implements and everything else keeps the scalar function. The integer-dot matmuls and the
// activation quantization are bit-identical to the scalar kernels by construction, the float
// matmuls differ by reduction order within a measured, locked tolerance.

#include "kernels/kernels.h"

namespace axon::kernels::neon {

// Fills `table`, which starts as a copy of the scalar table, with the NEON overrides.
void install(KernelTable& table);

// One per file.
void install_float(KernelTable& table);
void install_quant(KernelTable& table);
void install_q6k(KernelTable& table);
void install_attention_f32(KernelTable& table);
void install_attention_f16(KernelTable& table);

}  // namespace axon::kernels::neon
