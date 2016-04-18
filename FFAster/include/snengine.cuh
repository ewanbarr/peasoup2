
#ifndef FFASTER_SNENGINE_H_
#define FFASTER_SNENGINE_H_

#include "ffaster.h"
#include "base.cuh"

namespace FFAster
{
  namespace Kernels
  {
      __device__ __forceinline__
      float max_reduce_k_simple(float* primary,
				float* secondary,
				unsigned int size,
				unsigned int nlayers)
      {
	  float max_val = primary[0];
	  for (int ii=1;ii<size;ii++)
	      max_val = fmaxf(primary[ii],max_val);
	  return max_val;
      }


    //Inline so needs to be in header
    __device__ __forceinline__
    float max_reduce_k(float* primary,
		       float* secondary,
		       unsigned int size,
		       unsigned int nlayers)
    {
      unsigned int k;
      float* swap_ptr;
      int x, idx;
      // Power of two max reduction within a block                                           
      // Loop over ceil(log2(size))                                                          
      for (int ii=0; ii<nlayers; ii++)
        {
          k = 1<<ii;
          // Calculate max of idx and idx+k (wrapping in size)                               
          // for all idx and store in tmp array                                              
          for (idx=threadIdx.x; idx<size; idx+=blockDim.x)
            {
	      x = idx+k;
              if (x<size)
		secondary[idx] = fmaxf(primary[idx],primary[x]);
	      else
		secondary[idx] = primary[idx];
            }
          swap_ptr = primary;
          primary = secondary;
          secondary = swap_ptr;
	  __syncthreads();
        }
      // element 0 should now be the maximum                                                 
      return primary[0];
    }

      __device__ __forceinline__
      float max_reduce_k_2(float* primary,
			   float* secondary,
			   unsigned int size,
			   unsigned int nlayers)
      {
	  int idx;
	  int warp_id = threadIdx.x/32;
	  int lane_id = threadIdx.x & 31;
	  float max_val = 0.0;

	  secondary[threadIdx.x] = 0.0;
	  
	  for (idx=threadIdx.x; idx<size; idx+=blockDim.x){
	      max_val = fmaxf(primary[idx],max_val);
	  }
	  __syncthreads();
	  
	  //Now do a warp max reduce for all active warps
	  for (int ii=16; ii>=1; ii/=2){
	      if (ii+threadIdx.x < blockDim.x)
		  max_val = fmaxf(max_val,__shfl_down(max_val,ii));
	      __syncthreads();
	  }
	  	  
	  if (lane_id==0)
	      secondary[warp_id] = max_val;
	  __syncthreads();
	  
	  if (warp_id!=0) return 0.0;
	  max_val = secondary[lane_id];
	  __syncthreads();
	  
	  for (int ii=16; ii>=1; ii/=2){
	      if (ii+threadIdx.x < blockDim.x)
		  max_val = fmaxf(max_val,__shfl_down(max_val,ii));
	      __syncthreads();
	  }
	  return max_val;
      }
          
    __global__
    void matched_filter_max_k(float* input,
			      ffa_output_t* output,
			      unsigned int size,
			      unsigned int nwidths,
			      unsigned int nlayers,
			      float period = 0.0,
			      float pstep = 0.0);
      
  };
  
  template <class TransformType>
  class FFAOutputAnalyser :public TransformType
  {
  protected:
    float max_width_fraction;
    
  public:
    FFAOutputAnalyser(float max_width_fraction_=0.25)
      :max_width_fraction(max_width_fraction_){}
    
    virtual void execute(float* input,
			 ffa_output_t* output,
			 ffa_params_t& plan)=0;
    
    virtual void execute(float* input,
			 ffa_output_t* output,
			 size_t xdim,
			 size_t ydim,
			 float period,
			 float pstep)=0;
    
    virtual size_t get_required_output_bytes(ffa_params_t& plan)
    {
      return plan.nturns_pow2 * sizeof(ffa_output_t);
    }
    virtual size_t get_required_tmp_bytes(ffa_params_t& plan){return 0;}
  };

  template <class TransformType=Base::DeviceTransform>
  class MatchedFilterAnalyser: public FFAOutputAnalyser<TransformType>
  {
  public:
    MatchedFilterAnalyser(float max_width_fraction_=0.25)
      :FFAOutputAnalyser<TransformType>(max_width_fraction_){}
    
    size_t get_required_tmp_bytes(ffa_params_t& plan){return 0;}

    void execute(float* input,
		 ffa_output_t* output,
		 ffa_params_t& plan)
    {
      execute(input, output, plan.period_samps, plan.nturns_pow2, plan.period, plan.pstep);
    }
    
    void execute(float* input,
		 ffa_output_t* output,
		 size_t xdim,
		 size_t ydim,
		 float period,
		 float pstep);
  };
    
};

#include "detail/snengine.inl"

#endif
