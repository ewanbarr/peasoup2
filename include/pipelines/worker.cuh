#ifndef PEASOUP_WORKER_CUH
#define PEASOUP_WORKER_CUH

#include <vector>
#include <thread>
#include "cuda.h"

#include "utils/nvtx.hpp"
#include "data_types/timeseries.cuh"
#include "data_types/candidates.cuh"
#include "pipelines/args.hpp"
#include "utils/utils.cuh"
#include "utils/printer.hpp"

namespace peasoup {
    namespace pipeline {
	
	template <typename QueueType>
	class WorkerBase
	{
	protected:
	    QueueType& queue;
	    
	public:
	    WorkerBase(QueueType& queue):queue(queue){}	    
	    virtual void prepare()=0;
	    virtual void run()=0;
	    virtual void set_stream(cudaStream_t stream)=0;
	};


	/* There should only be one worker pool per context. A pool cannot
	   service more than one context as all of its associate buffers and 
	   transforms are only allocated on the initial context.
	*/
	template <typename WorkerType, typename QueueType, typename ArgsType>
	class WorkerPool
	{
	private:
	    std::vector<std::thread> threads;
	    std::vector<WorkerType*> workers;
	    std::vector<cudaStream_t> streams;
	    QueueType& queue;
	    ArgsType& args;
	    unsigned nworkers;

	public:
	    WorkerPool(QueueType& queue, ArgsType& args, unsigned nworkers);
	    ~WorkerPool();
	    void prepare();
	    void run();
	    void join();
	};
    } //worker
} //peasoup

#include "pipelines/detail/worker.inl"
 
#endif //PEASOUP_WORKER_CUH
