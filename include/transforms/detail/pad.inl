#include "transforms/pad.cuh"

namespace peasoup {
    namespace transform {

	template <System system, typename T>
	inline float Pad<system,T>::get_mean()
	{
	    float mean = thrust::reduce(this->get_policy(),input.data.begin(),input.data.begin()+original_size);
	    return mean/original_size;
	}

	template <System system, typename T>
	inline Pad<system,T>::Pad(type::TimeSeries<system,T>& input,
				  size_t original_size,
				  size_t padded_size)
	    :input(input),original_size(original_size),
	     padded_size(padded_size){}

	template <System system, typename T>
        inline void Pad<system,T>::prepare()
	{
	    LOG(logging::get_logger("transforms.pad"),logging::DEBUG,
		"Preparing data padder\nPadding from ",
		original_size," to ",padded_size," samples.\n",
		"Input metadata: \n",input.metadata.display(),
		"Input size: ",input.data.size()," samples");
	    input.data.resize(padded_size);
	}

	template <System system, typename T>
        inline void Pad<system,T>::execute()
	{
	    float mean = get_mean();
	    LOG(logging::get_logger("transforms.pad"),logging::DEBUG,
		"Padding with mean of ",mean);
	    thrust::fill(this->get_policy(),input.data.begin()+original_size,input.data.end(),mean);
	}
    } // transform
} // peasoup

