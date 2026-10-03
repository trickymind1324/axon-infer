add_executable(axon main.cpp)
target_link_libraries(axon PRIVATE axon::engine CLI11::CLI11)
target_compile_definitions(axon PRIVATE
  AXON_BENCH_TEXT="${PROJECT_SOURCE_DIR}/tests/reference/fixtures/perplexity/wikitext2-test-head.txt")
axon_configure_target(axon)
