# Need cmake version >= 3.9 for FindMPI 
cmake_minimum_required(VERSION 3.9)

# Need this for cmake >= 3.12
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(parallel_tasks)

find_package(MPI COMPONENTS C REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
