add_executable(axon main.cpp)
target_link_libraries(axon PRIVATE axon::model CLI11::CLI11)
axon_configure_target(axon)
