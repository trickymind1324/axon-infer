#include "kernels/neon/neon.h"

namespace axon::kernels::neon {

void install(KernelTable& table) {
  static_cast<void>(table);  // overrides arrive one file at a time
}

}  // namespace axon::kernels::neon
