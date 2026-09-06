# Interface target carrying the warning set. Linked only by project targets, so
# -Werror can never fail on a fetched dependency.
add_library(axon_warnings INTERFACE)

target_compile_options(
  axon_warnings
  INTERFACE -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            -Wimplicit-fallthrough)

# Newer clang flags __COUNTER__ (used by Catch2's macros) as a C2y extension under -Wpedantic.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wc2y-extensions AXON_HAS_WC2Y_EXTENSIONS)
if(AXON_HAS_WC2Y_EXTENSIONS)
  target_compile_options(axon_warnings INTERFACE -Wno-c2y-extensions)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(axon_warnings INTERFACE -Wduplicated-cond -Wlogical-op -Wuseless-cast)
endif()
