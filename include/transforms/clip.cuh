#ifndef PEASOUP_CLIP_CUH
#define PEASOUP_CLIP_CUH

#include <thrust/transform.h>
#include "data_types/timeseries.cuh"
#include "misc/policies.cuh"
#include "transforms/transform_base.cuh"
#include "utils/logging.hpp"

namespace peasoup {
    namespace transform {
	namespace functor {

	    template <typename T>
            struct clipper: thrust::unary_function<T,T>
            {
                float thresh;
                clipper(float thresh);

		__host__ __device__
		T operator()(T in) const;
            };
	    
	}

	template <System system, typename T>
	class Clip: public Transform<system>
	{
	private:
	    type::TimeSeries<system,T>& input;
	    type::TimeSeries<system,T>& output;
	    float thresh;
	    
	public:
	    Clip(type::TimeSeries<system,T>& input, 
		type::TimeSeries<system,T>& output,
		float thresh);
	    void prepare();
	    void execute();
	};
	
    } //transform
} //peasoup

#include "transforms/detail/clip.inl"

#endif //PEASOUP_CLIP_CUH
