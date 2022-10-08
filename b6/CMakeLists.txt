cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(detection)

find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(classify main.cpp)
target_link_libraries(classify PRIVATE torch)
