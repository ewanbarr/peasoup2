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
		float val = input[idx];

		if (nharms>0){
		    val += input[__float_as_int(__int_as_float(idx)*0.5f)];
		    output[idx] = val;
		}
		
		if (nharms>1){
                    #pragma unroll
		    for (ii=1;ii<4;ii+=2)
			val += input[__float_as_int(__int_as_float(idx)*ii/4.0f)];
		    output[size+idx] = val;
		}
		if (nharms>2){
                    #pragma unroll
		    for (ii=1;ii<8;ii+=2)
			val += input[__float_as_int(__int_as_float(idx)*ii/8.0f)];
		    output[2*size+idx] = val;
		}
		if (nharms>3){
                    #pragma unroll
		    for (ii=1;ii<16;ii+=2)
			val += input[__float_as_int(__int_as_float(idx)*ii/16.0f)];
		    output[3*size+idx] = val;
		}
                if (nharms>4){
                    #pragma unroll
                    for (ii=1;ii<32;ii+=2)
			val += input[__float_as_int(__int_as_float(idx)*ii/32.0f)];
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
                    val += input[(int)(idx*0.5f)];
                    output[idx] = val;
                }

                if (nharms>1){
                    #pragma unroll
                    for (ii=1;ii<4;ii+=2)
                        val += input[(int)(idx*ii/4.0f)];
                    output[size+idx] = val;
                }
                if (nharms>2){
                    #pragma unroll
                    for (ii=1;ii<8;ii+=2)
                        val += input[(int)(idx*ii/8.0f)];
                    output[2*size+idx] = val;
                }
                if (nharms>3){
                    #pragma unroll
                    for (ii=1;ii<16;ii+=2)
                        val += input[(int)(idx*ii/16.0f)];
                    output[3*size+idx] = val;
                }
                if (nharms>4){
                    #pragma unroll
                    for (ii=1;ii<32;ii+=2)
                        val += input[(int)(idx*ii/32.0f)];
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
        void HarmonicSum<system,T>::execute()
	{
	    utils::print(__PRETTY_FUNCTION__,"\n");
	    thrust::counting_iterator<size_t> begin(0);
	    thrust::counting_iterator<size_t> end = begin + input.data.size();
	    thrust::for_each(this->get_policy(), begin, end,functor::harmonic_sum<T>
			     (input_ptr,output_ptr,nharms,input.data.size()));
	}

	template <>
        inline void HarmonicSum<DEVICE,float>::execute()
        {
	    int nthreads = 512;
	    int nblocks = input.data.size()/nthreads + 1;
	    kernel::harmonic_sum_kernel<<<nblocks,nthreads>>>(input_ptr, output_ptr,(unsigned int)input.data.size(),nharms);
	}

    } //transform
} //peasoup
