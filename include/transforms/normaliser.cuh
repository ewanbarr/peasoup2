#ifndef PEASOUP_NORMALISER_CUH
#define PEASOUP_NORMALISER_CUH

#include "thrust/functional.h"
#include "thrust/complex.h"

#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"

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
	
	class NormaliserBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void normalise()=0;
	};
	
	template <System system, typename T>
	class Normaliser: public NormaliserBase
	{
	private:
	    type::FrequencySeries<system, thrust::complex<T> >& input;
	    type::FrequencySeries<system, thrust::complex<T> >& output;
	    type::FrequencySeries<system, T>& baseline;
	    SystemPolicy<system> policy_traits;
	    
	public:
	    Normaliser(type::FrequencySeries<system, thrust::complex<T> >& input,
		       type::FrequencySeries<system, thrust::complex<T> >& output,
		       type::FrequencySeries<system, T>& baseline)
		:input(input),output(output),baseline(baseline) {}
	    void prepare();
	    void normalise();
	};
	
    }
}

#include "transforms/detail/normaliser.inl"

#endif //PEASOUP_NORMALISER_CUH
