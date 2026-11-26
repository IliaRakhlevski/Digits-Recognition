cmake_minimum_required(VERSION 3.16)
project(DigitsRecognition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(digits_recognition Source.cpp)
target_link_libraries(digits_recognition PRIVATE ${OpenCV_LIBS})
