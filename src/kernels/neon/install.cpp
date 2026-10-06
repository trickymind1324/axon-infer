#include "kernels/neon/neon.h"

namespace axon::kernels::neon {

void install(KernelTable& table) {
  install_float(table);
}

}  // namespace axon::kernels::neon
