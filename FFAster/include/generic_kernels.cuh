#ifndef FFASTER_GENERIC_KERNELS_CUH_
#define FFASTER_GENERIC_KERNELS_CUH_

#include "ffaster.h"

namespace FFAster
{
  namespace Kernels
  {
    __global__
    void multiply_by_value_k(float *input,
			     float *output,
			     size_t size,
			     float value);
  };
  
  void multiply_by_value(float *input,
			 float *output,
			 size_t size,
			 float value,
			 cudaStream_t stream);
  
};

#include "detail/generic_kernels.inl"

#endif
