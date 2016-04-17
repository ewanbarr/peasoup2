
#include "snengine.cuh"

using namespace FFAster;

__global__
void Kernels::matched_filter_max_k(float* input,
				   ffa_output_t* output,
				   unsigned int size,
				   unsigned int nwidths,
				   unsigned int nlayers,
				   float period,
				   float pstep)
{
  extern __shared__ float shared[];
  float * primary = shared;
  float * secondary = primary+size;
  float * max_primary = secondary+size;
  float * max_secondary = max_primary+size;
  float * swap_ptr;
  float val,max_val,new_max;
  int width;
  unsigned int idx,shift;
  unsigned int row = blockIdx.x;

  // Each block handles one "row" of the input                                                             
  // Here we point the block at the start of                                                               
  // the row.                                                                                              
  input = input+row*size;
  output = output+row;
  
  // The length of a row may be longer than MAX_THREADS                                                    
  // so we loop over the block dimension to read all data                                                  
  // into shared memory                                                                                    
  for (idx=threadIdx.x; idx<size; idx+=blockDim.x)
    {
      val = input[idx];
      primary[idx] = val;
      max_primary[idx] = val;
    }
  __syncthreads();
  
  //get max with no filtering first                                                                        
  max_val = max_reduce_k(max_primary,max_secondary,size,nlayers);
  width = 1;
  /*
  if (threadIdx.x == 0)
    output[0] = max_val;
  */


  // This is a power of two cyclical sum reduction                                                         
  // for the whole row.                                                                                    
  for (int ii=0; ii<nwidths-1;ii++)
    {
      // calculate the jump to the next data point to be read                                              
      shift = 1<<ii;

      // for each element in row, calculate the summation                                                  
      // and store in a different shared memory area to                                                    
      // avoid muchos problemos with lack of synchronisation                                               
      // between different iterations of the loop                                                          
      for (idx=threadIdx.x; idx<size; idx+=blockDim.x)
	{
	  val = (primary[idx]+primary[(idx+shift)%size])/SQRT2;
	  max_primary[idx] = val;
	  secondary[idx] = val;
	}
      swap_ptr = primary;
      primary = secondary;
      secondary = swap_ptr;
      __syncthreads();

      // Determine the maximum value of the row                                                            
      // using a power of 2 parallel reduction                                                             
      new_max = max_reduce_k(max_primary,max_secondary,size,nlayers);
      
      if (new_max > max_val)
	{
	  max_val = new_max;
	  width = shift*2;
	}
    }
  if (threadIdx.x ==0)
    {
      output->snr = max_val;
      output->width = width;
      output->period = period+blockIdx.x*pstep;
    }
}

template<> 
void MatchedFilterAnalyser<Base::DeviceTransform>::execute
(float* input,
 ffa_output_t* output,
 size_t xdim,
 size_t ydim,
 float period,
 float pstep)
{
  int nblocks = ydim;
  int nthreads = min((int)xdim,(int)MAX_THREADS);
  int shared_space = 4 * sizeof(float) * xdim;
  int nwidths = (int) log2(xdim*max_width_fraction);
  int nlayers = (int) ceil(log2((float)xdim));
  Kernels::matched_filter_max_k<<<nblocks,nthreads,shared_space,stream>>>
    (input,output,xdim,nwidths,nlayers,period,pstep);
  Utils::check_cuda_error("FFAOutputAnalyser::operator() error from matched_filter_max_k",stream);
}
		
template<>
size_t MatchedFilterAnalyser<Base::HostTransform>::get_required_tmp_bytes(ffa_params_t& plan)
{
  return plan.period_samps * sizeof(float);
}
								 
template<>
void MatchedFilterAnalyser<Base::HostTransform>::execute
(float* input,
 ffa_output_t* output,
 size_t xdim,
 size_t ydim,
 float period,
 float pstep)
{
  float* tmp = (float*) this->tmp_storage;
  int nwidths = (int) ceil( log2( xdim * max_width_fraction ) );
  for (int ii=0; ii< ydim; ii++)
    {
      int offset = ii*xdim;
      ffa_output_t *tmp_output = output+ii;
      float new_max;
      float max_val = *std::max_element(input+offset,input+offset+xdim);
      int width = 1;

      for (int jj=1; jj< nwidths; jj++)
	{
	  int step = 1<<jj;
	  for (int kk=0; kk< xdim; kk++)
	    {
	      tmp[kk] = 0;
	      for (int ll=0; ll< step; ll++)
		{
		  tmp[kk]+=input[offset+(kk+ll)%xdim];
		}
	    }
	  new_max = *std::max_element(tmp,tmp+xdim)/sqrtf(step);
	  if (new_max>max_val)
	    {
	      max_val = new_max;
	      width = step;
	    }
	}
      tmp_output->snr = max_val;
      tmp_output->width = width;
    }
}
