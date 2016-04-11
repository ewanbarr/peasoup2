#ifndef PEASOUP_PAD_CUH
#define PEASOUP_PAD_CUH

#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "cuda.h"

#include "data_types/timeseries.cuh"
#include "misc/policies.cuh"
#include "transforms/transform_base.cuh"
#include "utils/logging.hpp"

namespace peasoup {
    namespace transform {

	template <System system, typename T>
	class Pad: public Transform<system>
	{
	private:
	    type::TimeSeries<system,T>& input;
	    size_t original_size;
	    size_t padded_size;
	    float get_mean();
	    
	public:
	    Pad(type::TimeSeries<system,T>& input, size_t original_size, size_t padded_size);
	    void prepare();
	    void execute();
	};
	
    } //transform
} //peasoup

#include "transforms/detail/pad.inl"

#endif //PEASOUP_PAD_CUH
