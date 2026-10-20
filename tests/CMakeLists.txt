include(Catch)

file(GLOB AXON_UNIT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp)

add_executable(axon_tests ${AXON_UNIT_TEST_SOURCES})
target_link_libraries(axon_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(axon_tests PRIVATE AXON_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/reference/fixtures")
if(AXON_BUILD_SERVER)
  target_link_libraries(axon_tests PRIVATE axon::server)  # brings the engine with it
  target_compile_definitions(axon_tests PRIVATE AXON_HAVE_SERVER=1)
else()
  target_link_libraries(axon_tests PRIVATE axon::engine)
endif()
axon_configure_target(axon_tests)

catch_discover_tests(axon_tests)

if(AXON_FUZZ)
  add_subdirectory(fuzz)
endif()

# Opt-in tests against downloaded models (skipped unless AXON_MODELS_DIR is set).
file(GLOB AXON_INTEGRATION_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp)
add_executable(axon_integration_tests ${AXON_INTEGRATION_TEST_SOURCES})
target_link_libraries(axon_integration_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(axon_integration_tests PRIVATE AXON_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/reference/fixtures")
if(AXON_BUILD_SERVER)
  target_link_libraries(axon_integration_tests PRIVATE axon::server)  # brings the engine with it
  target_compile_definitions(axon_integration_tests PRIVATE AXON_HAVE_SERVER=1)
else()
  target_link_libraries(axon_integration_tests PRIVATE axon::engine)
endif()
axon_configure_target(axon_integration_tests)
catch_discover_tests(axon_integration_tests PROPERTIES LABELS integration TIMEOUT 3600)
