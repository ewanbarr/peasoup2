#ifndef PEASOUP_NORMALISER_CUH
#define PEASOUP_NORMALISER_CUH

#include "thrust/functional.h"
#include "thrust/complex.h"

#include "misc/system.cuh"
#include "transforms/transform_base.cuh"
#include "data_types/frequencyseries.cuh"
#include "utils/logging.hpp"

namespace peasoup {
    namespace transform {
	namespace functor{
	    
	    template <typename T>
	    struct power_normalise: 
		public thrust::binary_function<thrust::complex<T>,thrust::complex<T>,thrust::complex<T> >
	    {
		inline __host__ __device__ 
		thrust::complex<T> operator()(thrust::complex<T> input, T local_median) const;
	    };
	    
	} // namespace functor
	
	template <System system, typename T>
	class Normaliser: public Transform<system>
	{
	private:
	    type::FrequencySeries<system, thrust::complex<T> >& input;
	    type::FrequencySeries<system, thrust::complex<T> >& output;
	    type::FrequencySeries<system, T>& baseline;
	    
	public:
	    Normaliser(type::FrequencySeries<system, thrust::complex<T> >& input,
		       type::FrequencySeries<system, thrust::complex<T> >& output,
		       type::FrequencySeries<system, T>& baseline)
		:input(input),output(output),baseline(baseline) {}
	    void prepare();
	    void execute();
	};
	
    }
}

#include "transforms/detail/normaliser.inl"

#endif //PEASOUP_NORMALISER_CUH
