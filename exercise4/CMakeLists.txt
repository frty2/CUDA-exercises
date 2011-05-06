cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(YAMLCPP REQUIRED)
include_directories(${YAMLCPP_INCLUDE_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories("src")

#cpu version
configure_file(src/raytracer.cu src/raytracer.cc COPYONLY)
add_executable(rtcpu src/main.cc src/parser.cc src/raytracer.cc src/ppm_writer.cc src/types.cc)
set_target_properties(rtcpu PROPERTIES COMPILE_FLAGS "-D__CPUVERSION__")
target_link_libraries(rtcpu ${GLOG_LIBRARY})
target_link_libraries(rtcpu ${GFLAGS_LIBRARY})
target_link_libraries(rtcpu ${YAMLCPP_LIBRARY})

#cuda version
cuda_add_executable(rtgpu src/main.cc src/parser.cc src/raytracer.cu src/ppm_writer.cc src/types.cc OPTIONS "-D__GPUVERSION__")
set_target_properties(rtgpu PROPERTIES COMPILE_FLAGS "-D__GPUVERSION__")
target_link_libraries(rtgpu ${GLOG_LIBRARY})
target_link_libraries(rtgpu ${GFLAGS_LIBRARY})
target_link_libraries(rtgpu ${YAMLCPP_LIBRARY})

# common flags
SET(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas -Wno-long-long -Wno-variadic-macros")
SET(CMAKE_CXX_FLAGS_RELEASE "-DNEXPENSIVE -O2 -g0")
SET(CMAKE_CXX_FLAGS_DEBUG "-DWARNINGS -O0 -g3 -fno-inline")

