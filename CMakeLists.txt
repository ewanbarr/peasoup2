# project name and specify the minumum cmake version.
project(peasoup)
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# cmake setup.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})

find_package(Thrust REQUIRED)
include_directories(${THRUST_INCLUDE_DIR})

find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDES})

find_package(Dedisp REQUIRED)
include_directories(${DEDISP_INCLUDES})

find_package(NVTX REQUIRED)
include_directories(${NVTX_INCLUDES})

# project version.
# set(VERSION_MAJOR 0)
# set(VERSION_MINOR 0)
# set(VERSION_PATCH 0)
# set(PROJECT_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH} ${PROJECT_SOURCE_VERSION}")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)

# Set the build type if it isn't already
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# === set compiler defaults
include(compiler_settings)

# === set up cuda from cuda.cmake
include(cuda)

# === set up default cuda flags
#set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_20,code=sm_20")
#set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_25,code=sm_25")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_30,code=sm_30")
#set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_35,code=sm_35")
#set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50,code=sm_50")

# === set up project includes
include_directories(include)

# === process through the peasoup tree:
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmarking)

# === set up FFAster
add_subdirectory(FFAster)
