#ifndef PEASOUP_BASELINEFINDER_CUH
#define PEASOUP_BASELINEFINDER_CUH

#include <cmath>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include "data_types/timeseries.cuh"
#include "misc/constants.h"
#include "misc/system.cuh"
#include "transforms/transform_base.cuh"
#include "utils/logging.hpp"

namespace peasoup {
    namespace transform {
	
	template <typename T> inline __host__ __device__ T median3(T a, T b, T c); 
	template <typename T> inline __host__ __device__ T median4(T a, T b, T c, T d); 
	template <typename T> inline __host__ __device__ T median5(T a, T b, T c, T d, T e); 
    
	namespace functor {
	    template <typename T>
	    struct median5_functor: public thrust::unary_function<T,T> 
	    {
		const T* in;

		median5_functor(const T* in_): in(in_) {}

		inline __host__ __device__ T operator()(unsigned int i) const;
	    };
	    
	    template <typename T>
	    struct linear_stretch_functor: public thrust::unary_function<T,T>
            {
                const T* in;
                unsigned in_size;
		float step;
		float correction;
		
                linear_stretch_functor(const T* in_, unsigned in_size, float step)
                    : in(in_), in_size(in_size), step(step), correction(((int)(step/2))/step){}

                inline __host__ __device__ T operator()(unsigned out_idx) const;
            };
	} // namespace functor
	

	template <System system, typename T>
	class BaselineFinder: public Transform<system>
	{
	protected:
	    typedef typename SystemVector<system,T>::vector_type vector_type;
	    std::vector< vector_type > medians;
	    void median_scrunch5(const vector_type& in, vector_type& out);
            void linear_stretch(const vector_type& in, vector_type& out, float step);

	public:
	    const std::vector< vector_type >& get_medians() {return medians;};
	    virtual void prepare()=0;
            virtual void execute()=0;
	};
	
	template <System system, typename T>
	class FDBaselineFinder: public BaselineFinder<system,T>
	{
	private:
	    typedef typename SystemVector<system,T>::vector_type vector_type;
	    type::FrequencySeries< system,T >& input;
	    type::FrequencySeries< system,T >& output;
	    std::vector< size_t > boundaries;
	    vector_type intermediate;
	    float accel_max;
	    
	public:
	    FDBaselineFinder(type::FrequencySeries< system,T >& input,
			     type::FrequencySeries< system,T >& output,
			     float accel_max=500.0)
		:input(input),output(output),accel_max(accel_max){}
	    const std::vector< size_t >& get_boundaries() {return boundaries;};
	    void prepare();
	    void execute();
	};

	template <System system, typename T>
        class TDBaselineFinder: public BaselineFinder<system,T>
        {
        private:
	    typedef typename SystemVector<system,T>::vector_type vector_type;
	    type::TimeSeries< system,T >& input;
	    type::TimeSeries< system,T >& output;
	    float smoothing_interval; //in seconds
	    
        public:
            TDBaselineFinder(type::TimeSeries< system,T >& input,
                             type::TimeSeries< system,T >& output,
			     float smoothing_interval)
                :input(input),output(output),
		 smoothing_interval(smoothing_interval){}
            void prepare();
            void execute();
        };

	

    } //transform
} //peasoup

#include "transforms/detail/baselinefinder.inl"

#endif // PEASOUP_BASELINEFINDER_CUH

