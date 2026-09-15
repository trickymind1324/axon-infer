file(GLOB_RECURSE AXON_KERNEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_kernels STATIC ${AXON_KERNEL_SOURCES})
add_library(axon::kernels ALIAS axon_kernels)
target_include_directories(axon_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_kernels PUBLIC axon::core)
axon_configure_target(axon_kernels)
