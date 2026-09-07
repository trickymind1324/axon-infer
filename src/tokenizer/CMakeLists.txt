file(GLOB AXON_TOKENIZER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_tokenizer STATIC ${AXON_TOKENIZER_SOURCES})
add_library(axon::tokenizer ALIAS axon_tokenizer)
target_include_directories(axon_tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_tokenizer PUBLIC axon::core)
axon_configure_target(axon_tokenizer)
