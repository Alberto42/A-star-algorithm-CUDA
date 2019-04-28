cmake_minimum_required(VERSION 3.13)
project(Project1b)

LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu/")

set(CMAKE_CXX_STANDARD 17)

add_executable(Project1b main.cpp)
TARGET_LINK_LIBRARIES(Project1b libboost_program_options.a)
