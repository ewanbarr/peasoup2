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
	    LOG(logging::get_logger("transform.spectrumformer"),logging::DEBUG,
                "Preparing SpectrumFormer\n",
		"Using nearest neighbour method: ",nn,"\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	    output.metadata.nn = nn;
	    LOG(logging::get_logger("transform.spectrumformer"),logging::DEBUG,
		"Prepared SpectrumFormer\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
	}
	
	template <System system, typename T>
	void SpectrumFormer<system,T>::execute()
	{ 
	    if (nn){
		LOG(logging::get_logger("transform.spectrumformer"),logging::DEBUG,
		    "Executing nearest neighbour spectrum forming");
		thrust::transform(this->get_policy(),input.data.begin()+1,
				  input.data.end(),input.data.begin(),output.data.begin()+1,
				  functor::interpolate_spectrum<T>());
	    } else {
		LOG(logging::get_logger("transform.spectrumformer"),logging::DEBUG,
                    "Executing basic spectrum forming");
		thrust::transform(this->get_policy(),input.data.begin(),
				  input.data.end(),output.data.begin(),
				  functor::complex_abs<T>());
	    }
	    output.data[0] = 0;
	}
		
    } //transform
} //peasoup


