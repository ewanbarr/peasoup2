#ifndef PEASOUP_HARMONICSUM_CUH
#define PEASOUP_HARMONICSUM_CUH

#include "cuda.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include <thrust/execution_policy.h>
#include "transforms/transform_base.cuh"
#include "utils/printer.hpp"

texture<float, 1, cudaReadModeElementType> harmonic_sum_texture;

namespace peasoup {
    namespace transform {
	namespace kernel {
	    static __global__ void harmonic_sum_kernel(float* d_odata, unsigned int size, unsigned int nharms);
	}

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
	
	
	template <System system, typename T>
	class HarmonicSum: public Transform<system>
	{
	private:
	    type::FrequencySeries<system,T>& input;
	    type::HarmonicSeries<system,T>& output;
	    unsigned nharms;
	    T* input_ptr;
	    T* output_ptr;
	    bool use_default;

	public:
	    HarmonicSum(type::FrequencySeries<system,T>& input, 
			type::HarmonicSeries<system,T>& output,
			unsigned nharms,
			bool use_default=false)
		:input(input),output(output),nharms(nharms),use_default(use_default){}
	    void _default_execute();
	    void prepare();	    
	    void execute();
	};
	
    } //transform
} //peasoup

#include "transforms/detail/harmonicsum.inl"

#endif // PEASOUP_HARMONICSUM_CUH

