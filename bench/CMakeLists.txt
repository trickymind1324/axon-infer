add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE axon::core benchmark::benchmark_main)
axon_configure_target(bench_kernels)

add_executable(bench_tokenizer bench_tokenizer.cpp)
target_link_libraries(bench_tokenizer PRIVATE axon::tokenizer benchmark::benchmark_main)
target_compile_definitions(bench_tokenizer PRIVATE AXON_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/reference/fixtures")
axon_configure_target(bench_tokenizer)
