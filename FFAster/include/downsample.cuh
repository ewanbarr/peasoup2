

#ifndef FFASTER_DOWNSAMPLE_CUH_
#define FFASTER_DOWNSAMPLE_CUH_

#include "ffaster.h"
#include "factorise.cuh"

namespace FFAster {
  namespace Kernels {
    
    __device__ __inline__
    float warp_sum_k(float val, 
		     const size_t factor);
    
    __global__
    void warp_downsample_k(const float *input,
			   float *output,
			   const size_t factor,
			   const size_t size,
			   const size_t out_size,
			   const size_t logical_warp_size,
			   const size_t logical_block_size,
			   const size_t n_per_block,
			   const size_t n_per_warp);

  }; /* namespace Kernels */
  
  
  class CachedDownsampler 
  {
  public:
    CachedDownsampler* parent;
    std::map<unsigned int, CachedDownsampler*> cache;
    float *data;
    size_t size;
    unsigned int downsampled_factor;
    Factoriser* factoriser;
    Allocators::ScratchAllocator *allocator;
    unsigned int max_factor;
    bool dummy;
    
    CachedDownsampler(float* data,
		      size_t size,
		      unsigned int max_factor=32,
		      bool dummy=false);

    CachedDownsampler(CachedDownsampler* parent,
		      float* data, 
		      size_t size,
		      unsigned int downsampled_factor,
		      bool dummy=false);

    ~CachedDownsampler();
    
    void set_allocator(Allocators::ScratchAllocator* allocator);
    
    size_t get_required_bytes();
    
    unsigned int closest_factor(unsigned int factor);
    
    CachedDownsampler* downsample(unsigned int factor);
    
  };

  
  void downsample_up_to_32(const float* data,
                        const size_t size,
                        float* output,
			unsigned int factor);
  
}; /* namespace FFAster */

#endif

