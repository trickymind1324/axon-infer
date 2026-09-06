add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE axon::core benchmark::benchmark_main)
axon_configure_target(bench_kernels)
