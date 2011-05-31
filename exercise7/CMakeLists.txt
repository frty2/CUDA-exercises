cmake_minimum_required(VERSION 2.8.0)

# for finding GLog
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIR})

find_package(GLog REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

include_directories("src")

set(CPU_SRC 
    src/mandelbrot.cc src/ppm_writer.cc src/mainwindow.cc)
    
#cpu version
add_executable(mandelbrot ${CPU_SRC})
target_link_libraries(mandelbrot ${GLOG_LIBRARY})
target_link_libraries(mandelbrot ${GLUT_LIBRARY})
target_link_libraries(mandelbrot ${OPENGL_gl_LIBRARY})
target_link_libraries(mandelbrot ${OPENCL_LIBRARIES})
