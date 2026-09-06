include(Catch)

file(GLOB AXON_UNIT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp)

add_executable(axon_tests ${AXON_UNIT_TEST_SOURCES})
target_link_libraries(axon_tests PRIVATE axon::core Catch2::Catch2WithMain)
axon_configure_target(axon_tests)

catch_discover_tests(axon_tests)
