#include "pipelines/fft_based/accelsearch_worker.cuh"

namespace peasoup {
    namespace pipeline {
	
	template <System system, typename QueueType>
	AccelSearchWorker<system,QueueType>::AccelSearchWorker(QueueType& queue, AccelSearchArgs& args)
	    :WorkerBase<QueueType>(queue),args(args)
	{
	    LOG(logging::get_logger("pipeline.accelsearchworker"),logging::DEBUG,
		"Creating accelsearch worker");
	    input.data.resize(this->queue.get_nsamps());
	    input.metadata.tsamp = this->queue.get_tsamp();
	    preproc = new Preprocessor<system>(input,input,args);
	    search = new AccelSearch<system>(input,dets,args);
	}
	
	template <System system, typename QueueType>
	AccelSearchWorker<system,QueueType>::~AccelSearchWorker()
	{
	    delete preproc;
	    delete search;
	}
	
	template <System system, typename QueueType>
	inline void AccelSearchWorker<system,QueueType>::prepare()
	{
	    LOG(logging::get_logger("pipeline.accelsearchworker"),logging::DEBUG,
                "Preparing accelsearch worker");
	    preproc->prepare();
	    search->prepare();
	}
	
	template <System system, typename QueueType>
	inline void AccelSearchWorker<system,QueueType>::set_stream(cudaStream_t stream)
	{
	    LOG(logging::get_logger("pipeline.accelsearchworker"),logging::DEBUG,
                "Setting stream on accelsearch worker");
	    preproc->set_stream(stream);
	    search->set_stream(stream);
	}
	
	template <System system, typename QueueType>
	inline void AccelSearchWorker<system,QueueType>::run()
	{
	    LOG(logging::get_logger("pipeline.accelsearchworker"),logging::DEBUG,
                "Start accelsearch worker consumer loop");
	    while (this->queue.pop(input)){
		LOG(logging::get_logger("pipeline.accelsearchworker"),logging::INFO,
		    "Processing timeseries with DM = ",input.metadata.dm," pccm");
		PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1);
		preproc->run();
		search->run();
		POP_NVTX_RANGE;
	    }
	}
    } //pipeline
} //peasoup
