#pragma once

#include <string>
#include <string_view>

namespace axon::unicode {

// Canonical composition (Unicode NFC). Text that is already NFC is returned unchanged; the
// common all-Latin-1 case is detected without allocating.
std::string normalize_nfc(std::string_view text);
bool is_nfc_fast_path(std::string_view text) noexcept;  // every byte < 0xCC: cannot need work

}  // namespace axon::unicode
