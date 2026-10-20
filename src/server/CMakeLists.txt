file(GLOB AXON_SERVER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_server STATIC ${AXON_SERVER_SOURCES})
add_library(axon::server ALIAS axon_server)
target_include_directories(axon_server PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_server PUBLIC axon::engine httplib::httplib nlohmann_json::nlohmann_json)
axon_configure_target(axon_server)
