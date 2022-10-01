cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(detection)

find_package(OpenCV REQUIRED COMPONENTS opencv_highgui)
find_package(Torch REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detection server.cpp)
target_link_libraries(detection PRIVATE opencv_highgui torch)

add_executable(detection_client client.cpp)
target_link_libraries(detection_client opencv_highgui)
