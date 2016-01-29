#ifndef PEASOUP_PAD_CUH
#define PEASOUP_PAD_CUH

#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "cuda.h"

#include "data_types/timeseries.cuh"
#include "misc/policies.cuh"
#include "transforms/transform_base.cuh"
#include "utils/printer.hpp"

namespace peasoup {
    namespace transform {

	template <System system, typename T>
	class Pad: public Transform<system>
	{
	private:
	    type::TimeSeries<system,T>& input;
	    size_t original_size;
	    size_t padded_size;
	    float get_mean()
            {
                float mean = thrust::reduce(this->get_policy(),input.data.begin(),input.data.begin()+original_size);
                return mean/original_size;
            }
	    
	public:
	    Pad(type::TimeSeries<system,T>& input,
		size_t original_size,
		size_t padded_size)
		:input(input),original_size(original_size),
		 padded_size(padded_size){}

	    void prepare(){
		utils::print(__PRETTY_FUNCTION__,"\n");
		input.metadata.display();
		input.data.resize(padded_size);
	    }
	    
	    void execute(){
		float mean = get_mean();
		thrust::fill(this->get_policy(),input.data.begin()+original_size,input.data.end(),mean);
	    }
	};
	
    }
}

#endif
