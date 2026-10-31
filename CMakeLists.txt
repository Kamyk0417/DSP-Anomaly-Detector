# Top-level build for the signal processing monorepo.
cmake_minimum_required(VERSION 3.20)

project(signal_proc LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Part 1 - C17 + SDL2 oscilloscope
add_subdirectory(oscilloscope)
