#include "transforms/spectrumformer.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {

	    template <typename T>
	    inline __host__ __device__
	    T complex_abs<T>::operator()(const thrust::complex<T> &x) const 
	    {
		T val = thrust::abs<T>(x);
		return val*val;
	    }

	    template <typename T>
	    inline __host__ __device__
	    T interpolate_spectrum<T>::operator()(const thrust::complex<T> &x, const thrust::complex<T> &y) const 
	    { 
		T val = thrust::max<T>(thrust::abs<T>(x-y)*RSQRT2,thrust::abs<T>(x)); 
		return val*val;
	    }
	    
	} // namespace functor
	
	template <System system, typename T>
	void SpectrumFormer<system,T>::prepare()
	{
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	}
	
	template <System system, typename T>
	void SpectrumFormer<system,T>::form_nn()
	{ 
	    output.data[0] = 0;
	    thrust::transform(policy_traits.policy,input.data.begin()+1,
			      input.data.end(),input.data.begin(),output.data.begin()+1,
			      functor::interpolate_spectrum<T>());
	    output.metadata.nn = true;
	}
	
	template <System system, typename T>
	void SpectrumFormer<system,T>::form()
	{
	    output.data[0] = 0;
	    thrust::transform(policy_traits.policy,input.data.begin(),
			      input.data.end(),output.data.begin(),
			      functor::complex_abs<T>());
	    output.metadata.nn = false;
	}
		
    } //transform
} //peasoup


