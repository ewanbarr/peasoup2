#include "transforms/clip.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    clipper<T>::clipper(float thresh):thresh(thresh){}

	    template <typename T>
	    __host__ __device__
	    T clipper<T>::operator()(T in) const {
		return in ? in < thresh : thresh; 
	    }

	}//namespace functor

	template <System system, typename T>
	inline Clip<system,T>::Clip(type::TimeSeries<system,T>& input,
				    type::TimeSeries<system,T>& output,
				    float thresh)
	    :input(input),output(output),thresh(thresh){}

	template <System system, typename T>
        inline void Clip<system,T>::prepare()
	{
	    LOG(logging::get_logger("transforms.clip"),logging::DEBUG,
		"Preparing data clipper\n",
		"Input metadata: \n",input.metadata.display(),
		"Input size: ",input.data.size()," samples");
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	}

	template <System system, typename T>
        inline void Clip<system,T>::execute()
	{
	    LOG(logging::get_logger("transforms.clip"),logging::DEBUG,
		"Clipping all data above ",thresh);
	    thrust::transform(this->get_policy(),input.data.begin(),
			      input.data.end(),output.data.begin(),
			      functor::clipper<T>(thresh));
	}
    } // transform
} // peasoup

