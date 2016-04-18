
#include "downsample.cuh"

using namespace FFAster;

/*
 * Kernels
 */

__device__ __forceinline__
float Kernels::warp_sum_k(float val,
			  const size_t factor)
{
  float out = val;
  for (size_t ii=1;ii<factor;ii++)
    out += __shfl_down(val,ii);
  return out;
}

__global__
void Kernels::warp_downsample_k(const float *input,
				float *output,
				const size_t factor,
				const size_t size,
				const size_t out_size,
				const size_t logical_warp_size,
				const size_t logical_block_size,
				const size_t n_per_block,
				const size_t n_per_warp)
{
  unsigned int lane_id = threadIdx.x&(WARP_SIZE-1);
  unsigned int warp_id = threadIdx.x/WARP_SIZE;
  unsigned int in_idx = blockIdx.x*logical_block_size + lane_id + warp_id*logical_warp_size;
  unsigned int out_idx = blockIdx.x*n_per_block + warp_id*n_per_warp + lane_id/factor;
  if (in_idx>=size)
    return;
  float val = input[in_idx];
  float out = warp_sum_k(val,factor);
  if (lane_id%factor==0 && lane_id<logical_warp_size)
    if (out_idx<out_size)
      output[out_idx] = out;
  return;
}

/*
 * CachedDownsampler methods
 */

CachedDownsampler::CachedDownsampler(float* data,
				     size_t size,
				     unsigned int max_factor,
				     bool dummy)
  :data(data),
   size(size),
   max_factor(max_factor),
   downsampled_factor(1),
   parent(NULL),
   allocator(NULL),
   dummy(dummy)
{
  if (max_factor>WARP_SIZE)
    throw std::runtime_error("maximum factorisation must be less than WARP_SIZE");
  factoriser = new Factoriser;
}

CachedDownsampler::CachedDownsampler(CachedDownsampler* parent,
				     float* data, 
				     size_t size,
				     unsigned int downsampled_factor,
				     bool dummy)
  :parent(parent),
   data(data),
   size(size),
   dummy(dummy),
   downsampled_factor(downsampled_factor)
{
  factoriser = parent->factoriser;
  max_factor = parent->max_factor;
  allocator  = parent->allocator;
}

void CachedDownsampler::set_allocator(Allocators::ScratchAllocator* allocator){
  this->allocator = allocator;
}

CachedDownsampler::~CachedDownsampler()
{
  typedef std::map<unsigned int, CachedDownsampler*>::iterator it_type;
  it_type iterator;
  for(iterator = cache.begin(); iterator != cache.end(); iterator++)
    delete iterator->second;
  
  if (parent != NULL && !dummy)
    {
      if (allocator == NULL)
	Utils::device_free(data);
      else
	allocator->device_free(data);
    }
  else if (parent != NULL && dummy) 
    {
      /*pass*/
    }
  else
    delete factoriser;
}

size_t CachedDownsampler::get_required_bytes()
{
  size_t total_size = 0;
  typedef std::map<unsigned int, CachedDownsampler*>::iterator it_type;
  it_type iterator;
  for(iterator = cache.begin(); iterator != cache.end(); iterator++)
    {
      total_size += iterator->second->size*sizeof(float);
      total_size += iterator->second->get_required_bytes();
    }
  return total_size;
}

unsigned int CachedDownsampler::closest_factor(unsigned int factor)
{
  return factoriser->get_nearest_factor(factor,max_factor);
}

CachedDownsampler* CachedDownsampler::downsample(unsigned int factor)
{
  if (factor==1)
    {
      return this;
    }
  else
    {
      if (factor != factoriser->get_nearest_factor(factor,max_factor))
	throw std::runtime_error("Downsampling factor must have a maximum factor less than 32");
      
      unsigned int first_factor = factoriser->first_factor(factor);
      if (cache.count(first_factor))
	{
	  return cache[first_factor]->downsample(factor/first_factor);
	}
      else
	{
	  float *downsampled_data = NULL;
	  size_t downsampled_size = size/first_factor;
	  	  	  
	  if (!dummy)
	    {
	      if (allocator == NULL)
		Utils::device_malloc<float>(&downsampled_data,downsampled_size);
	      else
		allocator->device_allocate<float>(&downsampled_data,downsampled_size);
	      downsample_up_to_32(data,size,downsampled_data,first_factor);
	      FFAster::Utils::check_cuda_error("Error from downsample_up_to_32");
	    }
	  cache[first_factor] = new CachedDownsampler(this,downsampled_data,downsampled_size,
						      downsampled_factor*first_factor,dummy);
	  return cache[first_factor]->downsample(factor/first_factor);
	}
    }
}

/*
 * Functions
 */

void FFAster::downsample_up_to_32(const float* data,
				  const size_t size,
				  float* output,
				  unsigned int factor)
{
  const size_t out_size = size/factor;
  if (out_size <= 0)
    throw std::runtime_error("Downsampling factor results in size of zero");
  if (factor > 32)
    throw std::runtime_error("Maximum downsampling factor is 32");
  size_t logical_warp_size = WARP_SIZE/factor * factor;
  size_t logical_block_size = logical_warp_size * MAX_THREADS/WARP_SIZE;
  size_t n_per_block = logical_block_size/factor;
  size_t n_per_warp = logical_warp_size/factor;
  size_t nblocks = (size_t) ceil(((float)size)/logical_block_size);

  Kernels::warp_downsample_k<<<nblocks,MAX_THREADS,0>>>(data,output, factor, size,
							out_size, logical_warp_size,
							logical_block_size, n_per_block,
							n_per_warp);
  return;
}

