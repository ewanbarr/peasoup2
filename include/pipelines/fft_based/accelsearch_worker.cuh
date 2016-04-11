#ifndef PEASOUP_ACCELSEARCHWORKER_CUH
#define PEASOUP_ACCELSEARCHWORKER_CUH

#include <vector>
#include "utils/nvtx.hpp"
#include "pipelines/fft_based/preprocessor.cuh"
#include "pipelines/fft_based/accelsearcher.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/candidates.cuh"
#include "pipelines/worker.cuh"
#include "pipelines/args.hpp"
#include "utils/utils.cuh"
#include "utils/logging.hpp"

namespace peasoup {
    namespace pipeline {
	
	template <System system, typename QueueType>
	class AccelSearchWorker: public WorkerBase<QueueType>
	{
        private:
	    type::TimeSeries<system,float> input;
	    std::vector< type::Detection > dets; 
	    Preprocessor<system>* preproc;
	    AccelSearch<system>* search;
	    AccelSearchArgs& args;
	    
        public:
	    AccelSearchWorker(QueueType& queue, AccelSearchArgs& args);
	    ~AccelSearchWorker();
	    void prepare();
	    void set_stream(cudaStream_t stream);
	    void run();
        };
	
    } //pipeline
} //peasoup

#include "pipelines/fft_based/detail/accelsearch_worker.inl"
 
#endif //PEASOUP_ACCELSEARCHWORKER_CUH
