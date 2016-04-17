#include "detrend.cuh"

using namespace FFAster;

/*
 * Kernels 
 */


__global__
void Kernels::median_of_5_reduce_k(float* input,
				   float* output,
				   int nout_per_block,
				   int pow_of_5,
				   size_t tot_size)
{
  extern __shared__ float shared[];
  const int n_per_block = blockDim.x * 5;
  const int offset = threadIdx.x * 5;
  const int global_offset = blockIdx.x * n_per_block;
  const int remaining = tot_size - global_offset;
  float * primary = shared;
  float * secondary = shared+n_per_block;
  float * swap_ptr;
  int size = min(n_per_block,remaining);
  int nreads = (int) ceil((float)size/blockDim.x);
  float a,b,c,d,e;
  
  // offset the input/output pointers
  input = input + global_offset;
  output = output + blockIdx.x * nout_per_block;

  for (int ii=0; ii<nreads; ii++)
    {
      int pos = threadIdx.x+ii*blockDim.x;
      if (pos<size)
	primary[pos] = input[pos];
    }
  __syncthreads();
  
  // for log5(N) steps perform median reduction
  for (int ii=0;ii<pow_of_5;ii++)
    {
      //cull unwanted threads
      if ((threadIdx.x+1)*5 > size)
        return;
      a = primary[offset];
      b = primary[offset+1];
      c = primary[offset+2];
      d = primary[offset+3];
      e = primary[offset+4];
      secondary[threadIdx.x] = median_of_5(a,b,c,d,e);
      swap_ptr = primary;
      primary = secondary;
      secondary = swap_ptr;
      size/=5;
      __syncthreads();
    }
  output[threadIdx.x] = primary[threadIdx.x];
  return;
}

__global__
void Kernels::remove_interpolated_baseline_k(float* input,
					     float* output,
					     float* medians,
					     size_t step,
					     size_t size,
					     size_t med_size)
{
  if (blockIdx.x > med_size-2)
    return;
  
  float x0, x1, y0, y1;
  unsigned int x0_idx = blockIdx.x;
  unsigned int x1_idx = x0_idx + 1;
  x0 = x0_idx*step + step/2.0;
  x1 = x1_idx*step + step/2.0;
  y0 = medians[x0_idx];
  y1 = medians[x1_idx];
  int step_by_2 = step/2;
  int start_idx,end_idx;

  if (x0_idx==0)
    {
      start_idx = threadIdx.x;
      end_idx = step+step_by_2;
    }
  else if (x0_idx == med_size-2)
    {
      start_idx = x0_idx*step + step_by_2 + threadIdx.x;
      end_idx = size;
    }
  else
    {
      start_idx = x0_idx*step + step_by_2 + threadIdx.x;
      end_idx = x1_idx*step + step_by_2;
    }
  float partial_interpolation = (y1-y0) / (x1-x0);
  for (int ii=start_idx; ii<end_idx; ii+=blockDim.x)
    {
      output[ii] = input[ii] - (y0 + partial_interpolation * (ii-x0));
    }
}


__global__
void Kernels::block_reduce_k(float *input,
			     float *output,
			     size_t size,
			     bool square)
{

  typedef cub::BlockReduce<float, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int num_valid = min((int)blockDim.x,(int)( size - blockDim.x * blockIdx.x ));
  
  if (idx>=size)
    return;
  float data = input[idx];
  if (square)
    data = data*data;
  float sum = BlockReduce(temp_storage).Sum(data,num_valid);
  if (threadIdx.x==0)
    output[blockIdx.x] = sum;
}


__global__
void Kernels::block_reduce_k_2(float *input,
			       float *output,
			       size_t size,
			       bool square)
{
  __shared__ float shared[32];
  
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  float value = 0;
  
  if (threadIdx.x < WARP_SIZE)
    shared[threadIdx.x] = 0;
  
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x & (WARP_SIZE-1);
  
  if (idx < size)
    value = input[idx];
  
  if (square)
    value = value * value;
  
  for (int ii=0; ii<5; ii++)
    value += __shfl_down(value,1<<ii);
  
  if (lane_id == 0)  
    shared[warp_id] = value;
  
  __syncthreads();

  if (threadIdx.x >= WARP_SIZE) 
    return;
  
  value = shared[threadIdx.x];
  
  __syncthreads();
  
  for (int ii=0; ii<5; ii++)
    value += __shfl_down(value,1<<ii);
  
  if (threadIdx.x == 0)
    output[blockIdx.x] = value;  
}

/*
 * MedianOfFiceBaselineEstimator methods
 */

template <> size_t MedianOfFiveBaselineEstimator<Base::DeviceTransform>::get_required_output_bytes(ffa_params_t& plan)
{
  return (int)(plan.downsampled_size/pow(5,power_of_5)) * sizeof(float);
}

template <> size_t MedianOfFiveBaselineEstimator<Base::DeviceTransform>::get_required_tmp_bytes(ffa_params_t& plan)
{
  size_t size = plan.downsampled_size;
  size_t tmp_bytes = 0;
  int npasses = (int) ceil(power_of_5/5.0);
  int current_power = power_of_5;
  int power;
  while (npasses > 0)
    {
      power = min(current_power,5);
      size/=pow(5,power);
      current_power-=power;
      npasses--;
      if (npasses != 0)
	tmp_bytes += size * sizeof(float);
    }
  return tmp_bytes;
}

template <> int MedianOfFiveBaselineEstimator<Base::DeviceTransform>::get_baseline_step()
{
  return pow(5,power_of_5);
}

template <> void MedianOfFiveBaselineEstimator<Base::DeviceTransform>::set_smoothing_length(size_t size)
{
  power_of_5 = (int) ceil(log(size)/log(5.0));
}

template <> void MedianOfFiveBaselineEstimator<Base::DeviceTransform>::execute(float *input, 
									       float* output, 
									       ffa_params_t& plan)
{
  size_t size = plan.downsampled_size;
  if ((size/(int)pow(5,power_of_5)) == 0)
    throw std::runtime_error
      ("MedianOfFiveBaselineEstimator::estimate_baseline "
       "zero output size");
  
  int npasses = (int) ceil(power_of_5/5.0);
  int current_power = power_of_5;
  int power;
  int nthreads = 625;
  int nreturn, nblocks, shared_space;
  float *tmp_input, *tmp_output;
  tmp_input = input;
 
  if (npasses == 1)
    tmp_output = output;
  else
    {
      if (tmp_storage_bytes==0 or tmp_storage==NULL)
	throw std::runtime_error
	  ("MedianOfFiveBaselineEstimator::estimate_baseline "
	   "internal memory required.");
      tmp_output = (float*) tmp_storage;
    }
  while (npasses > 0)
    {
      nblocks = ceil(size/powf(5.,5.));
      power = min(power_of_5,5);
      nreturn = pow(5,5-power);
      shared_space = 6 * nthreads * sizeof(float);
            
      Kernels::median_of_5_reduce_k<<<nblocks,nthreads,shared_space,stream>>>
      	(tmp_input,tmp_output,nreturn,power_of_5,size);
      size/=pow(5,power);
      current_power-=power;

      npasses--;
      tmp_input = tmp_output;
      if (npasses == 1)
	tmp_output = output;
      else
	tmp_output = tmp_output + size;
    }
  Utils::check_cuda_error("Error from median_of_5_reduce_k",stream);
}

/* StdDevNormaliser methods */

template <> float StdDevNormaliser<Base::DeviceTransform>::calculate_normalisation(float* input,
										   size_t size)
{
  float h_square_sum;
  float* tmp_in = (float*) this->tmp_storage;
  float* tmp_out = tmp_in + size/MAX_THREADS;
  float* swap;
  int nblocks;
  int nthreads = MAX_THREADS;
  bool square = true;
  int pass = 0;
  size_t n = size;

  while (true)
    {
      nblocks = (int) ceil(n/(float)nthreads);

      if (pass == 0)
	Kernels::block_reduce_k<<<nblocks,nthreads,0,stream>>>
	  (input,tmp_out,n,square);
      else
	Kernels::block_reduce_k<<<nblocks,nthreads,0,stream>>>
	  (tmp_in,tmp_out,n,square);
      square = false;
      pass++;
      swap = tmp_in;
      tmp_in = tmp_out;
      tmp_out = swap;
      if (nblocks == 1)
        break;
      n = nblocks;
    }
  
  cudaMemcpyAsync((void*)&h_square_sum, (void*)tmp_in,
		  sizeof(float), cudaMemcpyDeviceToHost, stream);
  Utils::check_cuda_error("StdDevNormaliser::calculate_normalisation "
			  "error from cudaMemcpyAsync",stream);
  return (float)sqrt(h_square_sum/size);
}

template <> size_t StdDevNormaliser<Base::DeviceTransform>::get_required_tmp_bytes(ffa_params_t& plan)
{
  size_t size = plan.downsampled_size;
  return 2 * ((int)ceil(size/(float)MAX_THREADS)) * sizeof(float);
}

template <> void StdDevNormaliser<Base::DeviceTransform>::execute(float* input,
                                                                  float* output,
                                                                  ffa_params_t& plan)
{
  size_t size = plan.downsampled_size;
  float factor = calculate_normalisation(input,size);
  FFAster::multiply_by_value(input,output,size,1./factor,stream);
}

/* LinearInterpBaselineSubtractor methods */
template <> void LinearInterpBaselineSubtractor<Base::DeviceTransform>::execute(float* input,
										float* output,
										ffa_params_t& plan)
{
  if (baseline_size<=1)
    throw std::runtime_error("LinearInterpBaselineSubtractor::execute "
			     "baseline size must be greater than 1");
  int nblocks = baseline_size-1;
  int nthreads = MAX_THREADS;
  Kernels::remove_interpolated_baseline_k<<<nblocks,nthreads,0,stream>>>
    (input,output,baseline,step_size,plan.downsampled_size,baseline_size);
  Utils::check_cuda_error("Error from remove_interpolated_baseline_k",stream);
}

template <> size_t LinearInterpBaselineSubtractor<Base::DeviceTransform>::get_required_output_bytes(ffa_params_t& plan)
{
  return plan.downsampled_size*sizeof(float);
}
