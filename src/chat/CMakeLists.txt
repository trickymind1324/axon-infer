file(GLOB AXON_CHAT_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(axon_chat STATIC ${AXON_CHAT_SOURCES})
add_library(axon::chat ALIAS axon_chat)
target_include_directories(axon_chat PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(axon_chat PUBLIC axon::tokenizer)
axon_configure_target(axon_chat)
