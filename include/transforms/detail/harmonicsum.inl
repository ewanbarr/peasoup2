#include "transforms/harmonicsum.cuh"

namespace peasoup {
    namespace transform {
	namespace kernel {
	    

	    static __global__ 
	    void harmonic_sum_kernel(float* input, float* output, unsigned int size, unsigned int nharms)
	    {
		const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if (idx>=size) return;
		int ii;
		//float val = input[idx];
		//float val = tex1Dfetch(harmonic_sum_texture,idx+0.5f);
		float val = tex1Dfetch(harmonic_sum_texture,idx+0.5f);

		if (nharms>0){
		    //val += input[(int)(idx*0.5f)];
		    //val += tex1Dfetch(harmonic_sum_texture,idx*0.5f+0.5f);
		    val += tex1Dfetch(harmonic_sum_texture,idx/2+0.5f);
		    output[idx] = val;
		}
		
		if (nharms>1){
                    #pragma unroll
		    for (ii=1;ii<4;ii+=2){
			//val += input[(int)(idx*ii/4.0f)];
			//val += tex1Dfetch(harmonic_sum_texture,idx*ii/4.0f+0.5f);
			val += tex1Dfetch(harmonic_sum_texture,ii*idx/4.0f+0.5f);
		    }
		    output[size+idx] = val;
		}
		if (nharms>2){
                    #pragma unroll
		    for (ii=1;ii<8;ii+=2){
			//val += input[(int)(idx*ii/8.0f)];
			//val += tex1Dfetch(harmonic_sum_texture,idx*ii/8.0f+0.5f);
			val += tex1Dfetch(harmonic_sum_texture,ii*idx/8.0f+0.5f);
		    }
		    output[2*size+idx] = val;
		}
		if (nharms>3){
                    #pragma unroll
		    for (ii=1;ii<16;ii+=2){
			//val += input[(int)(idx*ii/16.0f)];
			//val += tex1Dfetch(harmonic_sum_texture,idx*ii/16.0f+0.5f);
			val += tex1Dfetch(harmonic_sum_texture,ii*idx/16.0f+0.5f);
		    }
		    output[3*size+idx] = val;
		}
                if (nharms>4){
                    #pragma unroll
                    for (ii=1;ii<32;ii+=2){
			//val += input[(int)(idx*ii/32.0f)];
			//val += tex1Dfetch(harmonic_sum_texture,idx*ii/32.0f+0.5f);
			val += tex1Dfetch(harmonic_sum_texture,ii*idx/32.0f+0.5f);
		    }
                    output[4*size+idx] = val;
                }
	    }
	    
	}

	namespace functor {
	    
	    template <typename T>
	    inline __host__ __device__
	    void harmonic_sum<T>::operator() (unsigned idx) const
	    {
		int ii;
		T val = input[idx];
                if (nharms>0){
                    val = input[(int)(idx*0.5f+0.5f)] + val;
                    output[idx] = val;
                }

                if (nharms>1){
                    #pragma unroll
                    for (ii=1;ii<4;ii+=2)
                        val = input[(int)(idx*ii/4.0f+0.5f)] + val;
                    output[size+idx] = val;
                }
                if (nharms>2){
                    #pragma unroll
                    for (ii=1;ii<8;ii+=2)
                        val = input[(int)(idx*ii/8.0f+0.5f)] + val;
                    output[2*size+idx] = val;
                }
                if (nharms>3){
                    #pragma unroll
                    for (ii=1;ii<16;ii+=2)
                        val = input[(int)(idx*ii/16.0f+0.5f)] + val;
                    output[3*size+idx] = val;
                }
                if (nharms>4){
                    #pragma unroll
                    for (ii=1;ii<32;ii+=2)
                        val = input[(int)(idx*ii/32.0f+0.5f)] + val;
                    output[4*size+idx] = val;
                }
	    }
	} //namespace functor
		
	template <System system, typename T>
	void HarmonicSum<system,T>::prepare()
	{
	    utils::print(__PRETTY_FUNCTION__,"\n");
	    input.metadata.display();
	    output.data.resize(input.data.size()*nharms);
	    output.metadata.binwidths.clear();
	    for (int ii=0;ii<nharms;ii++)
		output.metadata.binwidths.push_back(input.metadata.binwidth/(1<<(ii+1)));
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    input_ptr = thrust::raw_pointer_cast(input.data.data());
	    output_ptr = thrust::raw_pointer_cast(output.data.data());
	    output.metadata.display();
	}
	

	template <System system, typename T>
	void HarmonicSum<system,T>::_default_execute()
	{
	    thrust::counting_iterator<size_t> begin(0);
	    thrust::counting_iterator<size_t> end = begin + input.data.size();
	    thrust::for_each(this->get_policy(), begin, end,functor::harmonic_sum<T>
                             (input_ptr,output_ptr,nharms,input.data.size()));
        }

	template <System system, typename T>
        void HarmonicSum<system,T>::execute()
	{
	    _default_execute();
	}

	template <>
        inline void HarmonicSum<DEVICE,float>::execute()
        {
	    if (use_default)
		_default_execute();
	    else {
		int nthreads = 1024; // hardwired for k40 occupancy
		int nblocks = input.data.size()/nthreads + 1;
		cudaBindTexture(0, harmonic_sum_texture, (void*)input_ptr, input.data.size()*sizeof(float));
		printf("Nblocks: %d, Nthreads: %d\n",nblocks,nthreads);
		kernel::harmonic_sum_kernel<<<nblocks,nthreads,0,this->stream>>>
		    (input_ptr, output_ptr,(unsigned int)input.data.size(),nharms);
		this->sync();
		cudaUnbindTexture(harmonic_sum_texture);
		
	    }
	}

    } //transform
} //peasoup
