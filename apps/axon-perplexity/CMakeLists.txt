add_executable(axon-perplexity main.cpp)
target_link_libraries(axon-perplexity PRIVATE axon::engine CLI11::CLI11)
target_compile_definitions(axon-perplexity PRIVATE
  AXON_PERPLEXITY_TEXT="${PROJECT_SOURCE_DIR}/tests/reference/fixtures/perplexity/wikitext2-test-head.txt")
axon_configure_target(axon-perplexity)
