file(GLOB AXON_SAMPLING_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_sampling STATIC ${AXON_SAMPLING_SOURCES})
add_library(axon::sampling ALIAS axon_sampling)
target_include_directories(axon_sampling PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_sampling PUBLIC axon::core)
axon_configure_target(axon_sampling)
