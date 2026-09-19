file(GLOB AXON_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_engine STATIC ${AXON_ENGINE_SOURCES})
add_library(axon::engine ALIAS axon_engine)
target_include_directories(axon_engine PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_engine PUBLIC axon::model axon::sampling axon::chat)
axon_configure_target(axon_engine)
