#ifndef PEASOUP_TF_FFT_PIPELINE_CUH
#define PEASOUP_TF_FFT_PIPELINE_CUH

#include <vector>
#include <utility>
#include "pipelines/dmtrialqueue.cuh"
#include "pipelines/fft_based/accelsearch_worker.cuh"
#include "data_types/timefrequency.cuh"
#include "data_types/dispersiontime.cuh"
#include "pipelines/args.hpp"
#include "utils/utils.cuh"
#include "transforms/dedisperser.cuh"

namespace peasoup {
    namespace pipeline {

	class TimeFrequencyFFTPipeline
	{
	private:
	    typedef pipeline::AccelSearchArgs args_type;
	    typedef typename type::DispersionTime<HOST,uint8_t> dmtrial_type;
	    typedef typename type::TimeFrequencyBits<HOST> tf_type;
	    typedef typename pipeline::DMTrialQueue<dmtrial_type> queue_type;
	    typedef typename pipeline::AccelSearchWorker<DEVICE, queue_type > worker_type;
	    typedef typename pipeline::WorkerPool<worker_type,queue_type,args_type> pool_type;
	    
	    tf_type& input;
	    dmtrial_type dmtrials;
	    transform::Dedisperser* dedisperser;
	    queue_type queue;
	    std::vector<pool_type*> pools;
	    int ngpus;
	    pipeline::AccelSearchArgs args;

	public:
	    TimeFrequencyFFTPipeline(tf_type& input, Options& opts);
	    ~TimeFrequencyFFTPipeline();
	    void prepare();
	    void run();
	};
    } // pipelines
} // peasoup

#include "pipelines/fft_based/detail/tf_fft_pipeline.inl"

#endif //PEASOUP_TF_FFT_PIPELINE_CUH
