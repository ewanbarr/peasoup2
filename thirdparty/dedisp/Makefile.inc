# The definitions in this file should be changed to match your system

# CUDA_PATH ?= /usr/local/cuda
# CUDA_PATH ?= /usr/local/cuda-5.0/
CUDA_PATH ?= /mnt/home/ebarr/Soft/CUDA7.0/

CUDA_DIR   = ${CUDA_PATH}
THRUST_DIR = ${CUDA_PATH}/include

# LIB_ARCH   = lib
LIB_ARCH   = lib64
# GPU_ARCH   = sm_13
GPU_ARCH   = sm_35
# GPU_ARCH   = sm_30

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
