cmake_minimum_required(VERSION 3.10)
project(yoloCarCounting)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(tracker src/main.cpp src/yolo_detector.cpp src/bytetrack.cpp src/kalman_filter.cpp src/utils.cpp)
target_include_directories(tracker PRIVATE include)
target_link_libraries(tracker ${OpenCV_LIBS})
