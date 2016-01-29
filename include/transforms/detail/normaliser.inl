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
	    utils::print(__PRETTY_FUNCTION__,"\n");
            input.metadata.display();
	    assert (input.data.size() == baseline.data.size());
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	    output.metadata.display();
	}
	
	template <System system, typename T>
	void Normaliser<system,T>::execute()
	{
	    thrust::transform(this->get_policy(), input.data.begin(),
			      input.data.end(), baseline.data.begin(),
			      output.data.begin(), functor::power_normalise<T>());
	}
    }
}
