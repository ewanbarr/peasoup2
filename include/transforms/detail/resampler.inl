#include "transforms/resampler.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    inline __host__ __device__
	    size_t acceleration_map::operator()(size_t x) const
	    { return (size_t)(x + x*accel_fact*(x-size)); }
	    
	} // namespace functor
	
	template <System system, typename T>
        inline void TimeDomainResampler<system,T>::set_accel(float accel)
	{
	    LOG(logging::get_logger("transform.resampler"),logging::DEBUG,
		"Setting accleration to ",accel," m/s/s");
	    this->accel=accel;
	}
	
	template <System system, typename T>
	void TimeDomainResampler<system,T>::prepare()
	{
	    LOG(logging::get_logger("transform.resampler"),logging::DEBUG,
                "Preparing resampler transform\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	    LOG(logging::get_logger("transform.resampler"),logging::DEBUG,
                "Prepared resampler transform\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
	}

	template <System system, typename T>
        void TimeDomainResampler<system,T>::execute()
	{
	    LOG(logging::get_logger("transform.resampler"),logging::DEBUG,
		"Resampling data to acceleration of ",accel," m/s/s");
	    double accel_fact = ((accel * input.metadata.tsamp) / (2 * SPEED_OF_LIGHT));
	    double size = input.data.size();
	    countit begin(0);
	    mapit iter(begin,functor::acceleration_map(accel_fact,size));
	    thrust::gather(this->get_policy(),
			   iter,iter+input.data.size(),
			   input.data.begin(),output.data.begin());
	}
    } //transform
} //peasoup


