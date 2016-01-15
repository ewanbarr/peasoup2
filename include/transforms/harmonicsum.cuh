#ifndef PEASOUP_HARMONICSUM_CUH
#define PEASOUP_HARMONICSUM_CUH

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include <thrust/execution_policy.h>

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    struct harmonic_sum 
	    {
		T* input;
		T* output;
		unsigned nharms;
		unsigned size;
		
		harmonic_sum(T* input, T* output, unsigned nharms, unsigned size)
		    :input(input),output(output),nharms(nharms),size(size) {}
		
		inline __host__ __device__ void operator() (unsigned idx) const;
	    };
	} //namespace functor
	
	
	class HarmonicSumBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void sum()=0;
	};
	
	template <System system, typename T>
	class HarmonicSum: public HarmonicSumBase
	{
	private:
	    type::FrequencySeries<system,T>& input;
	    type::HarmonicSeries<system,T>& output;
	    SystemPolicy<system> policy_traits;
	    unsigned nharms;
	    T* input_ptr;
	    T* output_ptr;
	    
	public:
	    HarmonicSum(type::FrequencySeries<system,T>& input, 
			type::HarmonicSeries<system,T>& output,
			unsigned nharms)
		:input(input),output(output),nharms(nharms){}
	    void prepare();	    
	    void sum();
	};
	
    } //transform
} //peasoup

#include "transforms/detail/harmonicsum.inl"

#endif // PEASOUP_RESAMPLER_CUH

