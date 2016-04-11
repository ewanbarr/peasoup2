#include "transforms/normaliser.cuh"
#include "misc/constants.h"
#include <assert.h>

namespace peasoup {
    namespace transform {
        namespace functor{
	    template <typename T>
	    inline __host__ __device__
	    thrust::complex<T> power_normalise<T>::operator()(thrust::complex<T> input, T local_median) const
	    {
		return sqrtf(LN4/local_median) * input;
	    }
	}
	
	template <System system, typename T>
	void Normaliser<system,T>::prepare()
	{
	    LOG(logging::get_logger("transform.normaliser"),logging::DEBUG,
		"Preparing normaliser transform\n",
		"Input metadata:\n",input.metadata.display(),
		"Input size: ",input.data.size()," samples\n",
		"Baseline metadata:\n",baseline.metadata.display(),
		"Baseline size: ",baseline.data.size()," samples");
	    
	    assert (input.data.size() == baseline.data.size());
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	    
	    LOG(logging::get_logger("transform.normaliser"),logging::DEBUG,
                "Prepared normaliser transform\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");	    
	}
	
	template <System system, typename T>
	void Normaliser<system,T>::execute()
	{
	    LOG(logging::get_logger("transform.normaliser"),logging::DEBUG,
                "Normalising data");
	    thrust::transform(this->get_policy(), input.data.begin(),
			      input.data.end(), baseline.data.begin(),
			      output.data.begin(), functor::power_normalise<T>());
	}
    } //transform
} // peasoup
