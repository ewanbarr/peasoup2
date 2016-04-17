#include "pipelines/dmtrialqueue.cuh"
#include "thrust/copy.h"

namespace peasoup {
    namespace pipeline {
	
	template <typename DMTimeType>
	template <typename TimeSeriesType>
	bool DMTrialQueue<DMTimeType>::pop(TimeSeriesType& timeseries)
	{
	    std::unique_lock<std::mutex> lock(this->d_mutex);
	    size_t nsamps = trials.get_nsamps();
	    if (idx<trials.metadata.dms.size()){
		iter start = trials.data.begin()+idx*nsamps;
		iter end = trials.data.begin()+nsamps*(idx+1);
		timeseries.data.resize(nsamps);
		timeseries.metadata.tsamp = trials.metadata.tsamp;
		timeseries.metadata.dm = trials.metadata.dms[idx];
		timeseries.metadata.acc = 0;
		thrust::copy(start,end,timeseries.data.begin());
		idx++;
		return true;
	    } else {
		return false;
	    }
            
	}
    } //pipeline
} //peasoup
