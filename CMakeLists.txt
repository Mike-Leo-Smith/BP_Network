cmake_minimum_required(VERSION 3.6)
project(BP_Network)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(SOURCE_FILES main.cpp Network/layer.cpp Network/layer.h Network/functor.h Network/sigmoid.cpp Network/sigmoid.h Network/network.cpp Network/network.h Network/data.cpp Network/data.h Network/mnist_loader.cpp Network/mnist_loader.h Network/functor.cpp)
add_executable(BP_Network ${SOURCE_FILES})