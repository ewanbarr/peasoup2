#ifndef PEASOUP_SPECTRUMFORMER_CUH
#define PEASOUP_SPECTRUMFORMER_CUH

#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/frequencyseries.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {

	    template <typename T>
	    struct complex_abs: public thrust::unary_function<T,thrust::complex<T> >
	    {
		inline __host__ __device__
		T operator()(const thrust::complex<T> &x) const;
	    }; 
	    
	    template <typename T>
            struct interpolate_spectrum: public thrust::binary_function<T, thrust::complex<T>, thrust::complex<T> >
            {
                inline __host__ __device__
                T operator()(const thrust::complex<T> &x, const thrust::complex<T> &y) const;
            }; 	
	    
	} // namespace functor
	
	class SpectrumFormerBase 
	{
	public:
	    virtual void prepare()=0;
	    virtual void form()=0;
	    virtual void form_nn()=0;
	};
	
	template <System system, typename T>
	class SpectrumFormer: public SpectrumFormerBase
	{
	private:
	    type::FrequencySeries<system, thrust::complex<T> >& input;
	    type::FrequencySeries<system,T>& output;
	    SystemPolicy<system> policy_traits;
	    
	public:
	    SpectrumFormer(type::FrequencySeries<system, thrust::complex<T> >& input, 
			   type::FrequencySeries<system,T>& output)
		:input(input),output(output) {}
	    
	    void prepare();
	    void form_nn();
	    void form();
	};
    } //transform
} //peasoup

#include "transforms/detail/spectrumformer.inl"

#endif // PEASOUP_SPECTRUMFORMER_CUH

