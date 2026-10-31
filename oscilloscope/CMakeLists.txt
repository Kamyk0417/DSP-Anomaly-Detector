# Part 1 - digital oscilloscope (C17 + SDL2)

find_package(SDL2 CONFIG QUIET)

if(NOT SDL2_FOUND)
    # Fallback for Windows setups without a system SDL2 installation.
    include(FetchContent)
    FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-2.30.9
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(SDL2)
endif()

add_executable(oscilloscope
    src/main.c
    src/adc.c
    src/generator.c
    src/window.c
    src/fft.c
    src/filter.c
    src/io.c
    src/features.c
    src/visualize.c
)

target_include_directories(oscilloscope PRIVATE include)

if(MSVC)
    target_compile_options(oscilloscope PRIVATE /W4 /WX)
else()
    target_compile_options(oscilloscope PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_link_libraries(oscilloscope PRIVATE SDL2::SDL2)
if(WIN32)
    target_link_libraries(oscilloscope PRIVATE SDL2::SDL2main)
endif()

add_custom_command(
    TARGET oscilloscope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/configs/default.json"
            "$<TARGET_FILE_DIR:oscilloscope>/default.json"
    COMMENT "Copying default config next to the executable"
)

# SDL2 may be a shared library (as built by the FetchContent fallback), so the
# runtime DLL has to sit next to the executable for it to start on Windows.
if(WIN32 AND TARGET SDL2::SDL2)
    add_custom_command(
        TARGET oscilloscope POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL2::SDL2>"
                "$<TARGET_FILE_DIR:oscilloscope>"
        COMMENT "Copying SDL2 runtime next to the executable"
    )
endif()
