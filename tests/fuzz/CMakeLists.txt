# libFuzzer targets. Only built with AXON_FUZZ=ON (the `fuzz` preset), which requires clang
# with compiler-rt's fuzzer runtime (Linux). Apple clang does not ship it.
add_executable(axon_fuzz_gguf fuzz_gguf.cpp)
target_link_libraries(axon_fuzz_gguf PRIVATE axon::core)
axon_configure_target(axon_fuzz_gguf)
target_compile_options(axon_fuzz_gguf PRIVATE -fsanitize=fuzzer)
target_link_options(axon_fuzz_gguf PRIVATE -fsanitize=fuzzer)
