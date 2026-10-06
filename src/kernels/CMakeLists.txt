file(GLOB AXON_KERNEL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ref/*.cpp)

# NEON files are compiled on arm64 only, with the dot product and half-precision extensions
# enabled for those files alone. They are reached only through the table that runtime
# detection selects, so the rest of the binary stays plain armv8.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  file(GLOB AXON_KERNEL_NEON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/neon/*.cpp)
  set_source_files_properties(${AXON_KERNEL_NEON_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+dotprod+fp16")
  list(APPEND AXON_KERNEL_SOURCES ${AXON_KERNEL_NEON_SOURCES})
endif()

add_library(axon_kernels STATIC ${AXON_KERNEL_SOURCES})
add_library(axon::kernels ALIAS axon_kernels)
target_include_directories(axon_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_kernels PUBLIC axon::core)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_compile_definitions(axon_kernels PUBLIC AXON_HAVE_NEON=1)
endif()
axon_configure_target(axon_kernels)
