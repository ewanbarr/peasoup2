#ifndef PEASOUP_PEAKFINDER_CUH
#define PEASOUP_PEAKFINDER_CUH

#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "data_types/candidates.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "misc/constants.h"
#include "misc/system.cuh"
#include "misc/policies.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    struct greater_than_threshold
		:thrust::unary_function< thrust::tuple<unsigned,T>, bool>
	    {
		float threshold;
		greater_than_threshold(float thresh)
		    :threshold(thresh){}
		inline __host__ __device__ 
		bool operator()(thrust::tuple<unsigned,T> t) const;
	    };
	    
	} // namespace functor

	class PeakFinderBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void execute()=0;
	};

	template <System system>
	class PeakFinderDerivedBase: public PeakFinderBase
        {
	public:
	    
        };

	template <>
	class PeakFinderDerivedBase<HOST>: public PeakFinderBase
        {
	protected:
	    thrust::detail::host_t policy;
	public:
	    PeakFinderDerivedBase():policy(thrust::host){}
        };

	template <>
	class PeakFinderDerivedBase<DEVICE>: public PeakFinderBase
	{
	protected:
	    thrust::detail::execute_with_allocator<policy::cached_allocator, thrust::system::cuda::detail::execute_on_stream_base> policy;
	    policy::cached_allocator allocator;
	public: 
	    PeakFinderDerivedBase():policy(thrust::cuda::par(allocator)){}
        };
	

	template <System system, typename T>
	class PeakFinder: public PeakFinderDerivedBase<system>
	{
	private:
	    typedef thrust::tuple<unsigned,T> peak;
	    type::FrequencySeries<system,T>& fundamental;
	    type::HarmonicSeries<system,T>& harmonics;
	    std::vector<type::Detection>& dets;
	    typename SystemVector<system,unsigned>::vector_type idxs;
	    typename SystemVector<system,float>::vector_type powers;
	    typename SystemVector<HOST,unsigned>::vector_type h_idxs;
            typename SystemVector<HOST,float>::vector_type h_powers;
	    typedef typename SystemVector<system,unsigned>::vector_type::iterator idx_iter;
	    typedef typename SystemVector<system,T>::vector_type::iterator pow_iter;
	    typedef thrust::tuple< thrust::counting_iterator<unsigned>, pow_iter > peak_tuple_in;
	    typedef thrust::tuple< idx_iter, pow_iter > peak_tuple_out;
	    float minsigma;
	    std::vector<float> thresholds;
	    void _execute(pow_iter in, size_t size, int nh, float df, float thresh);
	    
	public:
	    PeakFinder(type::FrequencySeries<system,T>& fundamental,
		       type::HarmonicSeries<system,T>& harmonics,
		       std::vector<type::Detection>& dets,
		       float minsigma)
		:PeakFinderDerivedBase<system>(),fundamental(fundamental),
		 harmonics(harmonics),dets(dets),minsigma(minsigma){}
	    
	    std::vector<float>& get_thresholds();
	    void prepare();
	    void execute();
	};
	
    } //transform
} //peasoup

#include "transforms/detail/peakfinder.inl"

#endif // PEASOUP_PEAKFINDER_CUH

