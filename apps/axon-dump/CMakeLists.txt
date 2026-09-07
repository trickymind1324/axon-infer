add_executable(axon-dump main.cpp)
target_link_libraries(axon-dump PRIVATE axon::core CLI11::CLI11)
axon_configure_target(axon-dump)
