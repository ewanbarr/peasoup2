#include "generic_kernels.cuh"

using namespace FFAster;

__global__
void Kernels::multiply_by_value_k(float *input,
                                  float *output,
                                  size_t size,
                                  float value)
{
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (idx>=size)
    return;
  output[idx] = input[idx]*value;
}

void FFAster::multiply_by_value(float *input,
                                float *output,
				size_t size,
                                float value,
				cudaStream_t stream)
{
  int nblocks = size/MAX_THREADS + 1;
  Kernels::multiply_by_value_k<<<nblocks,MAX_THREADS,0,stream>>>(input,output,size,value);
  Utils::check_cuda_error("Error from multiply_by_value kernel",stream);
}
