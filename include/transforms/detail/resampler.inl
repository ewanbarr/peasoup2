#include "transforms/resampler.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    inline __host__ __device__
	    size_t acceleration_map::operator()(size_t x) const
	    { return (size_t)(x + x*accel_fact*(x-size)); }
	    
	} // namespace functor
	
	template <System system, typename T>
	void TimeDomainResampler<system,T>::prepare()
	{
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	}

	template <System system, typename T>
        void TimeDomainResampler<system,T>::resample(float accel)
	{
	    double accel_fact = ((accel * input.metadata.tsamp) / (2 * SPEED_OF_LIGHT));
	    double size = input.data.size();
	    countit begin(0);
	    mapit iter(begin,functor::acceleration_map(accel_fact,size));
	    thrust::gather(policy_traits.policy,
			   iter,iter+input.data.size(),
			   input.data.begin(),output.data.begin());
	}
    } //transform
} //peasoup


