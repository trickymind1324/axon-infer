#include "kernels/neon/neon.h"

namespace axon::kernels::neon {

void install(KernelTable& table) {
  install_float(table);
  install_quant(table);
  install_q6k(table);
}

}  // namespace axon::kernels::neon
