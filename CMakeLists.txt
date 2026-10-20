cmake_minimum_required(VERSION 3.24)

project(
  axon-infer
  VERSION 0.0.1
  DESCRIPTION "LLM inference engine in C++20, built from scratch"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
# No C++ modules anywhere, so the dependency scan that newer CMake runs for C++20 Ninja builds
# is off: it needs clang-scan-deps, which minimal toolchains such as the Docker image lack.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AXON_BUILD_TESTS "Build unit tests" ON)
option(AXON_BUILD_BENCH "Build benchmarks" ON)
option(AXON_BUILD_SERVER "Build the HTTP server" ON)
option(AXON_NATIVE "Compile with -march=native (local peak numbers only)" OFF)
option(AXON_COVERAGE "Enable source-based coverage (clang only)" OFF)
option(AXON_CLANG_TIDY "Run clang-tidy on project targets" OFF)
option(AXON_FUZZ "Build libFuzzer targets (clang with compiler-rt fuzzer only)" OFF)
set(AXON_SANITIZE "" CACHE STRING "Comma-separated sanitizers: address,undefined,thread")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(warnings)
include(sanitizers)
include(deps)

# Git commit embedded into the version header. "unknown" outside a checkout.
find_package(Git QUIET)
set(AXON_GIT_COMMIT "unknown")
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE AXON_GIT_COMMIT_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE AXON_GIT_RESULT)
  if(AXON_GIT_RESULT EQUAL 0)
    set(AXON_GIT_COMMIT "${AXON_GIT_COMMIT_OUT}")
  endif()
endif()
configure_file(include/axon/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/axon/version.h @ONLY)

if(AXON_CLANG_TIDY)
  find_program(AXON_CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-17 clang-tidy-18 REQUIRED)
  # Only project headers are checked; fetched dependencies are never linted.
  set(AXON_CLANG_TIDY_COMMAND "${AXON_CLANG_TIDY_EXE}"
      "--header-filter=^${PROJECT_SOURCE_DIR}/(include|src|apps|bench|tests)/")
  if(APPLE)
    # A non-Apple clang-tidy needs the SDK path that Apple clang locates implicitly.
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE AXON_MACOS_SDK
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(AXON_MACOS_SDK)
      list(APPEND AXON_CLANG_TIDY_COMMAND "--extra-arg=-isysroot${AXON_MACOS_SDK}")
    endif()
  endif()
endif()

# Applies project-wide settings to one of our own targets (never to dependencies).
function(axon_configure_target target)
  target_link_libraries(${target} PRIVATE axon_warnings)
  target_compile_features(${target} PUBLIC cxx_std_20)
  if(AXON_NATIVE)
    target_compile_options(${target} PRIVATE -march=native)
  endif()
  if(AXON_COVERAGE)
    target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(${target} PRIVATE -fprofile-instr-generate)
  endif()
  if(AXON_CLANG_TIDY)
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${AXON_CLANG_TIDY_COMMAND}")
  endif()
  axon_apply_sanitizers(${target})
endfunction()

add_subdirectory(src)
add_subdirectory(apps)

if(AXON_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(AXON_BUILD_BENCH)
  add_subdirectory(bench)
endif()
