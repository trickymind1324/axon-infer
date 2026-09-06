file(GLOB AXON_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_core STATIC ${AXON_CORE_SOURCES})
add_library(axon::core ALIAS axon_core)

target_include_directories(
  axon_core
  PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/generated
         ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_core PUBLIC fmt::fmt)
find_package(Threads REQUIRED)
target_link_libraries(axon_core PUBLIC Threads::Threads)

axon_configure_target(axon_core)
