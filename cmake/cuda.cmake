cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})

# add sdk samples useful headerfiles like cuda_helpers.h
if(CUDA_SMP_INC)
  include_directories(${CUDA_SMP_INC})
endif(CUDA_SMP_INC)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Pass options to NVCC
list(APPEND CUDA_NVCC_FLAGS -DENABLE_CUDA --std c++11)
list(APPEND CUDA_NVCC_FLAGS_DEBUG --debug; --device-debug; --generate-line-info)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50) #Maxwell




