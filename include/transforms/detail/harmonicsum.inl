#include "transforms/harmonicsum.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    inline __host__ __device__
	    void harmonic_sum<T>::operator() (unsigned idx) const
	    {
		T val = input[idx];
		if (nharms>0)
		    {
			val += input[(unsigned) (idx*0.5 + 0.5)];
			output[idx] = val*rsqrt(2.0);
		    }
		
		if (nharms>1)
		    {
			val += input[(unsigned) (idx * 0.75 + 0.5)];
			val += input[(unsigned) (idx * 0.25 + 0.5)];
			output[size+idx] = val*0.5;
		    }
		
		if (nharms>2)
		    {
			val += input[(unsigned) (idx * 0.125 + 0.5)];
			val += input[(unsigned) (idx * 0.375 + 0.5)];
			val += input[(unsigned) (idx * 0.625 + 0.5)];
			val += input[(unsigned) (idx * 0.875 + 0.5)];
			output[2*size+idx] = val*rsqrt(8.0);
		    }
		
		if (nharms>3)
		    {
			val += input[(unsigned) (idx * 0.0625 + 0.5)];
			val += input[(unsigned) (idx * 0.1875 + 0.5)];
			val += input[(unsigned) (idx * 0.3125 + 0.5)];
			val += input[(unsigned) (idx * 0.4375 + 0.5)];
			val += input[(unsigned) (idx * 0.5625 + 0.5)];
			val += input[(unsigned) (idx * 0.6875 + 0.5)];
			val += input[(unsigned) (idx * 0.8125 + 0.5)];
			val += input[(unsigned) (idx * 0.9375 + 0.5)];
			output[3*size+idx] = val*0.25;
		    }
		
		if (nharms>4)
		    {
			val += input[(unsigned) (idx * 0.03125 + 0.5)];
			val += input[(unsigned) (idx * 0.09375 + 0.5)];
			val += input[(unsigned) (idx * 0.15625 + 0.5)];
			val += input[(unsigned) (idx * 0.21875 + 0.5)];
			val += input[(unsigned) (idx * 0.28125 + 0.5)];
			val += input[(unsigned) (idx * 0.34375 + 0.5)];
			val += input[(unsigned) (idx * 0.40625 + 0.5)];
			val += input[(unsigned) (idx * 0.46875 + 0.5)];
			val += input[(unsigned) (idx * 0.53125 + 0.5)];
			val += input[(unsigned) (idx * 0.59375 + 0.5)];
			val += input[(unsigned) (idx * 0.65625 + 0.5)];
			val += input[(unsigned) (idx * 0.71875 + 0.5)];
			val += input[(unsigned) (idx * 0.78125 + 0.5)];
			val += input[(unsigned) (idx * 0.84375 + 0.5)];
			val += input[(unsigned) (idx * 0.90625 + 0.5)];
			val += input[(unsigned) (idx * 0.96875 + 0.5)];
			output[4*size+idx] = val*rsqrt(32.0);
		    }
		return;
	    }
	} //namespace functor
		
	template <System system, typename T>
	void HarmonicSum<system,T>::prepare()
	{
	    output.data.resize(input.data.size()*nharms);
	    output.metadata.binwidths.clear();
	    for (int ii=0;ii<nharms;ii++)
		output.metadata.binwidths.push_back(input.metadata.binwidth/(1<<(ii+1)));
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    input_ptr = thrust::raw_pointer_cast(input.data.data());
	    output_ptr = thrust::raw_pointer_cast(output.data.data());
	}
	
	template <System system, typename T>
        void HarmonicSum<system,T>::sum()
	{
	    thrust::counting_iterator<size_t> begin(0);
	    thrust::counting_iterator<size_t> end = begin + input.data.size();
	    thrust::for_each(policy_traits.policy, begin, end,functor::harmonic_sum<T>
			     (input_ptr,output_ptr,nharms,input.data.size()));
	}
    } //transform
} //peasoup
