
#ifndef FFASTER_FFA_CUH_
#define FFASTER_FFA_CUH_

#include "ffaster.h"
#include "base.cuh"
#include "generic_kernels.cuh"

namespace FFAster
{

  namespace Kernels
  {

    __global__
    void ffa_k(float* input_layer,
	       float* output_layer,
	       const unsigned int layer,
	       const int size);

  }; /* namespace Kernels */

  
  template <class TransformType = Base::DeviceTransform>
  class FFA: public TransformType
  {
  public:
    virtual void execute(float* input,
			 float* output,
			 ffa_params_t& params)=0;
    virtual size_t get_required_output_bytes(ffa_params_t& params)=0;
    virtual size_t get_required_tmp_bytes(ffa_params_t& params)=0;
  };
  
  template <class TransformType = Base::DeviceTransform>
  class Radix2FFA: public FFA<TransformType>
  {
  private:
    float* buffer_a;
    float* buffer_b;
    
    void swap_tmp_buffers();

    void prepare(float* data,
		 ffa_params_t& params);
    
  public:
    void execute(float* input,
		 float* output,
		 ffa_params_t& params);
    size_t get_required_output_bytes(ffa_params_t& params);
    size_t get_required_tmp_bytes(ffa_params_t& params);
  };

}; /* namespace FFAster */

#include "detail/ffa.inl"

#endif
