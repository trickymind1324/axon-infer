add_executable(axon-server main.cpp)
target_link_libraries(axon-server PRIVATE axon::server CLI11::CLI11)
axon_configure_target(axon-server)
