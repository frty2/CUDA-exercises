cmake_minimum_required(VERSION 2.8.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories("src")

#cuda version
cuda_add_executable(dotproduct src/dotproduct.cu)

