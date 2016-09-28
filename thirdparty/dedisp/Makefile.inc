# The definitions in this file should be changed to match your system

# CUDA_PATH ?= /usr/local/cuda
# CUDA_PATH ?= /usr/local/cuda-5.0/
CUDA_PATH ?= /mnt/home/ebarr/Soft/CUDA7.0/

CUDA_DIR   = ${CUDA_PATH}
THRUST_DIR = ${CUDA_PATH}/include

# LIB_ARCH   = lib
LIB_ARCH   = lib64
# GPU_ARCH   = sm_13
# GPU_ARCH   = sm_20
GPU_ARCH   = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_30,code=sm_30

GCC        = gcc
GXX        = g++
AR         = ar
NVCC       = ${CUDA_PATH}/bin/nvcc
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

DEDISP_DEBUG = 0

#.SILENT :
