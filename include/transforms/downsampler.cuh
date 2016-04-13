#ifndef PEASOUP_DOWNSAMPLER_CUH
#define PEASOUP_DOWNSAMPLER_CUH

#include <map>

#include <thrust/transform.h>

#include "data_types/timeseries.cuh"
#include "misc/system.cuh"
#include "utils/factorise.hpp"
#include "utils/logging.hpp"

namespace peasoup {
    namespace transform {

	//Downsampling tree
	//Does not adhere to normal transform rules
	//Currently not implementing a CPU version

	namespace functor {
	    
	    template <typename T>
	    struct downsample: thrust::unary_function<T,unsigned int>
	    {
		T* input;
		unsigned int factor;
		unsigned int size;
		
		downsample(T* input, unsigned int size, unsigned int factor)
		    :input(input),factor(factor),size(size){}

		inline __host__ __device__ T operator()(unsigned int idx);
	    };

	}//functor

	template<System system, typename T>
	class CachedDownsampler
	{
	public:
	    CachedDownsampler<system,T>* parent;
	    std::map<unsigned, CachedDownsampler<system,T>*> cache;
	    type::TimeSeries<system,T>* data;
	    unsigned int downsampled_factor;
	    utils::Factoriser* factoriser;
	    unsigned int max_factor;
	    bool new_parent;

	    CachedDownsampler(type::TimeSeries<system,T>* data,
			      unsigned int max_factor=32);
	    CachedDownsampler(CachedDownsampler<system,T>* parent,
			      type::TimeSeries<system,T>* data,
			      unsigned int downsampled_factor);
	    ~CachedDownsampler();
	    unsigned int closest_factor(unsigned int factor);
	    CachedDownsampler* downsample(unsigned int factor);
	    void set_data(type::TimeSeries<system,T>* data);
	    void notify();
	};
	    
    } //transform
} //peasoup

#include "transforms/detail/downsampler.inl"

#endif //PEASOUP_PAD_CUH
