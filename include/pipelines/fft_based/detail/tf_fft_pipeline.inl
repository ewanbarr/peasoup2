#include "pipelines/fft_based/tf_fft_pipeline.cuh"

namespace peasoup {
    namespace pipeline {

	inline TimeFrequencyFFTPipeline::TimeFrequencyFFTPipeline(tf_type& input, Options& opts)
	    :input(input),queue(dmtrials)
	{
	    args.birdies.clear();
	    args.acc_list.clear();
	    for (int ii=0;ii<10;ii++)
		args.acc_list.push_back((float)ii);
	    args.birdies.push_back(std::pair<float,float>(123.0,0.2));
	    args.minsigma = opts.minsigma;
	    args.nharm = opts.nharm;
	    args.nfft = opts.nfft;
	    ngpus = std::min((int)utils::gpu_count(),(int)opts.ngpus);
	    dedisperser = new transform::Dedisperser(input,dmtrials,ngpus);
	    //if (opts.dm_list.size() == 0)
	    //dedisperser->gen_dmlist(opts.dm_start,opts.dm_end,opts.dm_pulse_width,opts.dm_tol);
	    //else
	    //dedisperser->set_dmlist(opts.dm_list);
	    dedisperser->gen_dmlist(0.0,10.0,40.0,1.05);
	    dedisperser->prepare();
	    
	    for (int n=0;n<ngpus;n++){
		cudaSetDevice(n);
		pools.push_back(new pool_type(queue,args,opts.nthreads));
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

