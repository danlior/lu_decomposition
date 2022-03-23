cmake_minimum_required(VERSION 3.10)
project(complete_lu_decomposition  VERSION 1.0
            DESCRIPTION "complete lu decomposition"
            LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# NOTE: the value of BUILD_SHARED_LIBS will be used to pick between STATIC and SHARED in add_library statements

# for now, assume Linux when setting compile option

#add_library(lu source/lu.cpp)

INCLUDE_DIRECTORIES(include/)
INCLUDE_DIRECTORIES(~/Eigen/eigen/)

enable_testing()
add_subdirectory(test)
