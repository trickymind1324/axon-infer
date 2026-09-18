file(GLOB AXON_MODEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_model STATIC ${AXON_MODEL_SOURCES})
add_library(axon::model ALIAS axon_model)
target_include_directories(axon_model PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_model PUBLIC axon::kernels axon::tokenizer)
axon_configure_target(axon_model)
