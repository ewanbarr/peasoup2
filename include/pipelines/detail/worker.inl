#include "pipelines/worker.cuh"

namespace peasoup {
    namespace pipeline {
	
	template <typename WorkerType, typename QueueType, typename ArgsType>
	WorkerPool<WorkerType,QueueType,ArgsType>::WorkerPool(QueueType& queue, ArgsType& args, unsigned nworkers)
	    :queue(queue),args(args),nworkers(nworkers)
	{
	    printf("Nbirds: %d\n",args.birdies.size());
	    for (unsigned ii=0;ii<nworkers;ii++)
		workers.push_back(new WorkerType(queue,args));
	}
	
	template <typename WorkerType, typename QueueType, typename ArgsType>
	WorkerPool<WorkerType,QueueType,ArgsType>::~WorkerPool()
	{
	    for (auto worker: workers)
		delete worker;
	    for (auto stream: streams)
		cudaStreamDestroy(stream);
	}
	
	template <typename WorkerType, typename QueueType, typename ArgsType>
	inline void WorkerPool<WorkerType,QueueType,ArgsType>::prepare()
	{
	    streams.resize(workers.size());
	    for (int ii=0;ii<workers.size();ii++){
		cudaStreamCreate(&streams[ii]);
		utils::check_cuda_error(__PRETTY_FUNCTION__);
		workers[ii]->set_stream(streams[ii]);
		workers[ii]->prepare();
	    }
	}
	
	template <typename WorkerType, typename QueueType, typename ArgsType>
	inline void WorkerPool<WorkerType,QueueType,ArgsType>::run()
	{
	    for (auto worker: workers){
		threads.push_back(std::thread(&WorkerType::run, worker));
	    }
	}
	
	template <typename WorkerType, typename QueueType, typename ArgsType>
	inline void WorkerPool<WorkerType,QueueType,ArgsType>::join()
	{
	    for (auto& t: threads)
		t.join();
	}
    } //worker
} //peasoup
 
