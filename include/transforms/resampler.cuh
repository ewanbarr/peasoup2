#ifndef PEASOUP_RESAMPLER_CUH
#define PEASOUP_RESAMPLER_CUH

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <thrust/gather.h>

#include "data_types/timeseries.cuh"
#include "misc/constants.h"
#include "misc/system.cuh"
#include "transforms/transform_base.cuh"
#include "utils/printer.hpp"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    struct acceleration_map: public thrust::unary_function<size_t,size_t>
	    {
		double accel_fact;
		double size;
		acceleration_map(double accel_fact,double size)
		    :accel_fact(accel_fact),size(size){};
		
		inline __host__ __device__ size_t operator()(size_t x) const;
	    };
	    
	} // namespace functor
	
	
	template <System system, typename T>
        class TimeDomainResampler: public Transform<system>
        {
        private:
	    typedef thrust::counting_iterator<size_t> countit;
	    typedef thrust::transform_iterator< functor::acceleration_map, countit > mapit;
	    type::TimeSeries< system,T >& input;
	    type::TimeSeries< system,T >& output;
	    float accel;

        public:
            TimeDomainResampler(type::TimeSeries< system,T >& input,
                                type::TimeSeries< system,T >& output)
                :input(input),output(output),accel(0){}
	    void prepare();
	    void set_accel(float accel){this->accel=accel;}
            void execute();
	};
    } //transform
} //peasoup

#include "transforms/detail/resampler.inl"

#endif // PEASOUP_RESAMPLER_CUH

