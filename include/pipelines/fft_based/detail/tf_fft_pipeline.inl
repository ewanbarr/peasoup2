#include <memory>
#include "pipelines/fft_based/tf_fft_pipeline.cuh"

namespace peasoup {
    namespace pipeline {

	inline TimeFrequencyFFTPipeline::TimeFrequencyFFTPipeline(tf_type& input, TimeFrequencyFFTPipelineArgs& args)
	    :input(input),queue(dmtrials),args(args)
	{
	    // TEMPORARY: This should be replaced with GPU selection
	    // This change requires an updated to dedisp and its interface
	    ngpus = std::min((int)utils::gpu_count(),(int)args.ngpus);

	    dedisperser = new transform::Dedisperser(input,dmtrials,ngpus);
	    if (args.dedispersion.dm_list.size()==0){
		dedisperser->gen_dmlist(args.dedispersion.dm_start,
					args.dedispersion.dm_end,
					args.dedispersion.dm_pulse_width,
					args.dedispersion.dm_tol);
	    }
	    else {
		dedisperser->set_dmlist(args.dedispersion.dm_list);
	    }
	    dedisperser->prepare();
	    
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		pools.push_back(new pool_type(queue,args.accelsearch,args.nthreads));
	    }
	}
	
	inline TimeFrequencyFFTPipeline::~TimeFrequencyFFTPipeline()
	{
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		delete pools[n];
	    }
	    delete dedisperser;
	}
	
	inline void TimeFrequencyFFTPipeline::prepare()
	{
	    
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		pools[n]->prepare();
	    }
	}
	
	inline void TimeFrequencyFFTPipeline::run()
	{
	    dedisperser->execute();
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		pools[n]->run();
		}
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		pools[n]->join();
		utils::check_cuda_error(__PRETTY_FUNCTION__);
	    }
	}
    } // pipeline
} // peasoup

