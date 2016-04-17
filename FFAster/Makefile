# 

NVCC = nvcc


# Output directories                                                                                                                        
BIN_DIR     = ./bin
OBJ_DIR     = ./obj
LIB_DIR     = ./lib

# Paths
SRC_DIR  = ./src
INCLUDE_DIR = ./include

# Compiler flags
OPTIMISE = -O3
DEBUG    = 

# Includes and libraries
INCLUDE  = -I$(INCLUDE_DIR) -I${CUDA_DIR}/include 
LIBS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lpthread 

# compiler flags
# --compiler-options -Wall
NVCC_COMP_FLAGS = -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35
NVCCFLAGS  = ${UCFLAGS} ${OPTIMISE} ${NVCC_COMP_FLAGS} --machine 64 -Xcompiler '-fPIC' ${DEBUG}
CFLAGS    = ${UCFLAGS} -fPIC ${OPTIMISE} ${DEBUG}

OBJECTS   = ffa snengine downsample detrend generic_kernels factorise

all: directories objects library

objects: 
	$(foreach var,$(OBJECTS), ${NVCC} ${NVCCFLAGS} ${INCLUDE} -dc ${SRC_DIR}/$(var).cu -o ${OBJ_DIR}/$(var).o ;) 

library:
	${NVCC} ${NVCCFLAGS} ${INCLUDE} -dlink $(wildcard ${OBJ_DIR}/*.o) -o ${OBJ_DIR}/link.o
	${NVCC} ${NVCCFLAGS} -shared -o ${LIB_DIR}/libffaster.so $(wildcard ${OBJ_DIR}/*.o) -lcudart

tests:
	${NVCC} ${UCFLAGS} ${OPTIMISE} ${NVCC_COMP_FLAGS} --machine 64 ${INCLUDE} -L${LIB_DIR} -lffaster src/tests/test_ffaplan.cu 

directories:
	@mkdir -p ${BIN_DIR}
	@mkdir -p ${OBJ_DIR}
	@mkdir -p ${LIB_DIR}

clean:
	@rm -rf ${BIN_DIR}/*	
	@rm -rf ${OBJ_DIR}/*
	@rm -rf ${LIB_DIR}/*