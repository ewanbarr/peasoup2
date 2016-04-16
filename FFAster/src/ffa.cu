#include "ffa.cuh"

using namespace FFAster;

/*
 * Kernels
 */

__global__
void Kernels::ffa_k(float* input_layer,
		    float* output_layer,
		    const unsigned int layer,
		    const int size)
{
  const unsigned int k = 1<<layer; 
  const unsigned int input_turn0 = blockIdx.x/k * k + blockIdx.x;
  const unsigned int input_turn1 = input_turn0 + k;
  const int shift0 = blockIdx.x&(k-1);
  const int shift1 = shift0 + 1;
  const unsigned int in_idx0 = size*input_turn0;
  const unsigned int in_idx1 = size*input_turn1;
  const unsigned int out_idx0 = (blockIdx.x*2)*size;
  const unsigned int out_idx1 = (blockIdx.x*2+1)*size;
  int idx,tmp;
  for (idx=threadIdx.x; idx<size; idx+=blockDim.x)
    {
      tmp = in_idx0+idx;
      output_layer[out_idx0+idx] = input_layer[tmp]+input_layer[in_idx1 + (idx + shift0)%size];
      output_layer[out_idx1+idx] = input_layer[tmp]+input_layer[in_idx1 + (idx + shift1)%size];
    } 
} 

/*
 * FFA methods
 */

template<> void Radix2FFA<Base::DeviceTransform>::prepare(float *data, 
					       ffa_params_t& params)
{
  buffer_a = (float*) tmp_storage;
  buffer_b = buffer_a + params.padded_size; 
  size_t valid_bytes = params.nturns * params.period_samps * sizeof(float);
  size_t redundant_turns = params.nturns_pow2 - params.nturns;
  size_t padding = redundant_turns * params.period_samps * sizeof(float);
  cudaMemcpyAsync((void*)buffer_a, (void*)data, valid_bytes,
		  cudaMemcpyDeviceToDevice, stream);
  void* ptr = (void*)(((char*)buffer_a) + valid_bytes);
  cudaMemsetAsync(ptr, 0, padding, stream);
  Utils::check_cuda_error("Radix2FFA::prepare error from Async operations",stream);
}

template<> void Radix2FFA<Base::DeviceTransform>::swap_tmp_buffers()
{
  float* swap_ptr = NULL;
  swap_ptr = buffer_a;
  buffer_a = buffer_b;
  buffer_b = swap_ptr;
}

template<> size_t Radix2FFA<Base::DeviceTransform>::get_required_output_bytes(ffa_params_t& params)
{
  return params.padded_size * sizeof(float);
}

template<> size_t Radix2FFA<Base::DeviceTransform>::get_required_tmp_bytes(ffa_params_t& params)
{
  return 2 * params.padded_size * sizeof(float);
}

template<> void Radix2FFA<Base::DeviceTransform>::execute(float* input,
					       float* output,
					       ffa_params_t& params)
{
  prepare(input,params);
  const int nblocks = params.nturns_pow2/2;
  const int nthreads = (int) min((int)params.period_samps,MAX_THREADS);
  for (int layer=0; layer<params.nlayers; layer++)
    {
      if (layer < params.nlayers-1)
	{
	  Kernels::ffa_k<<<nblocks,nthreads,0,stream>>>
	    (buffer_a,buffer_b,layer,(int)params.period_samps);
	  swap_tmp_buffers();
	}
      else
	Kernels::ffa_k<<<nblocks,nthreads,0,stream>>>
	  (buffer_a,output,layer,(int)params.period_samps);
    }
  Utils::check_cuda_error("Radix2FFA::execute error from Kernels::ffa_k",stream);
  FFAster::multiply_by_value(output,output,params.padded_size,1/sqrt(params.nturns),stream);
}
