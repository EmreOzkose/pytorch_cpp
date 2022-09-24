cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(detection)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detection main.cpp)

target_link_libraries(detection "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

set_property(TARGET detection PROPERTY CXX_STANDARD 14)
