# Qwen3-Omni TTS Integration Test Tool

add_executable(llama-qwen3omni-tts main.cpp)

target_link_libraries(llama-qwen3omni-tts PRIVATE
    llama
    common
    ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(llama-qwen3omni-tts PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/common
)

install(TARGETS llama-qwen3omni-tts RUNTIME)
