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

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    struct greater_than_threshold
		:thrust::unary_function< thrust::tuple<unsigned,T>, bool>
	    {
		float threshold;
		greater_than_threshold(float thresh):threshold(thresh){}
		inline __host__ __device__ 
		bool operator()(thrust::tuple<int,T> t) { 
		    return thrust::get<1>(t) > threshold; 
		}
	    };
	    
	} // namespace functor

	int device_find_peaks(int n, int start_index, float * d_dat,
			      float thresh, int * indexes, float * snrs,
			      thrust::device_vector<int>& d_index,
			      thrust::device_vector<float>& d_snrs,
			      cached_allocator& policy)
	{

	    using thrust::tuple;
	    using thrust::counting_iterator;
	    using thrust::zip_iterator;
	    typedef tuple< counting_iterator<unsigned>, decltype(input.data)::iterator > peak_tuple_in;
	    typedef tuple< decltype(idxs)::iterator, decltype(output.data)::iterator > peak_tuple_out;
	    counting_iterator<int> iter(0);
	    zip_iterator< peak_tuple_in > input_zip = make_zip_iterator(make_tuple(iter,input.data.begin()));
	    zip_iterator< peak_tuple_out > output_zip = make_zip_iterator(make_tuple(idxs.begin(),output.data.begin()));	    
	    
	    int num_copied = thrust::copy_if(thrust::cuda::par(policy), 
					     input_zip, 
					     input_zip+input.data.size(),
					     output_zip, 
					     greater_than_threshold<float>(thresh)
					     ) - output_zip;
	    
	    thrust::copy(d_index.begin(),d_index.begin()+num_copied,indexes);
	    thrust::copy(d_snrs.begin(),d_snrs.begin()+num_copied,snrs);
	    ErrorChecker::check_cuda_error("Error from device_find_peaks;");
	    return(num_copied);
	}

	class PeakFinderBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void execute()=0;
	};

	template <System system, typename T>
	class PeakFinder: public PeakFinderBase
	{
	private:
	    type::FrequencySeries<system,T>& input_f;
	    type::HarmonicSeriess<system,T>& input_h;
	    type::PeakList<system,T>& peaks;

	public:
	    PeakFinder(type::FrequencySeries<system,T>& input_f,
		       type::HarmonicSeriess<system,T>& input_h,
		       type::PeakList<system,T>& peaks)
		:input_f(input_f),input_h(input_h),peaks(peaks){}
	    
	    void prepare();
	    void execute();
	    
	};
	
    } //transform
} //peasoup

#include "transforms/detail/resampler.inl"

#endif // PEASOUP_RESAMPLER_CUH

