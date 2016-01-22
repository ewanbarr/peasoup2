#ifndef PEASOUP_DMTRIALQUEUE_CUH
#define PEASOUP_DMTRIALQUEUE_CUH

#include <mutex>

namespace peasoup {
    namespace pipeline {
	
	template <typename DMTimeType>
	class DMTrialQueue
	{
	private:
	    typedef typename DMTimeType::vector_type::iterator iter;
	    DMTimeType& trials; 
	    size_t idx;
	    std::mutex d_mutex;
	    
	public:
	    DMTrialQueue(DMTimeType& trials)
		:trials(trials),idx(0){}
	    
	    template <typename TimeSeriesType>
	    bool pop(TimeSeriesType& timeseries);
	};
    } //pipeline 
} //peasoup

#include "pipelines/detail/dmtrialqueue.inl"

#endif
