cmake_minimum_required(VERSION 3.23)
project(OpenCVLab)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(OpenCVLab main.cpp helpers/common.h helpers/common.cpp helpers/Functions.h helpers/Functions.cpp helpers/colorcode.h helpers/colorcode.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCVLab ${OpenCV_LIBS}) # replace binary_name
