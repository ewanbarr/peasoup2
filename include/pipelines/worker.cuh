#ifndef PEASOUP_WORKER_CUH
#define PEASOUP_WORKER_CUH

#include <vector>
#include <thread>

#include "utils/nvtx.hpp"
#include "pipelines/preprocessor.cuh"
#include "pipelines/accelsearcher.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/candidates.cuh"
#include "pipelines/args.hpp"

namespace peasoup {
    namespace pipeline {

	template <System system, typename QueueType>
	class AccelSearchWorker
	{
        private:
	    type::TimeSeries<system,float> input;
	    QueueType& queue;
	    std::vector< type::Detection > dets; 
	    Preprocessor<system>* preproc;
	    AccelSearch<system>* search;

        public:
	    AccelSearchWorker(QueueType& queue,
			      AccelSearchArgs& args)
		:queue(queue)
	    {
	
		input.data.resize(queue.get_nsamps());
		input.metadata.tsamp = queue.get_tsamp();
		preproc = new Preprocessor<system>(input,input,args);
		search = new AccelSearch<system>(input,dets,args);
	    }
	    
	    ~AccelSearchWorker()
	    {
		delete preproc;
		delete search;
	    }
	    
	    void prepare()
	    {
		preproc->prepare();
		search->prepare();
	    }

	    void run()
	    {
		while (queue.pop(input)){
		    printf("Processing timeseries with DM=%f\n",input.metadata.dm);
		    PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1);
		    preproc->run();
		    search->run();
		    POP_NVTX_RANGE
		}
	    }
        };

	template <typename WorkerType, typename QueueType, typename ArgsType>
	class WorkerPool
	{
	private:
	    std::vector<std::thread> threads;
	    std::vector<WorkerType*> workers;
	    QueueType& queue;
	    ArgsType& args;
	    unsigned nworkers;

	public:
	    WorkerPool(QueueType& queue, ArgsType& args, unsigned nworkers)
		:queue(queue),args(args),nworkers(nworkers)
	    {
		for (int ii=0;ii<nworkers;ii++)
                    workers.push_back(new WorkerType(queue,args));
	    }

	    ~WorkerPool()
	    {
		for (auto worker: workers)
                    delete worker;
	    }
	    
	    void prepare()
	    {
		for (auto worker: workers)
		    worker->prepare();
	    }
	    
	    void run()
	    {
		for (auto worker: workers)
                    threads.push_back(std::thread(&WorkerType::run, worker));
	    }

	    void join()
	    {
		for (auto& t: threads)
                    t.join();
	    }
	};
    } //worker
} //peasoup
 
#endif //PEASOUP_WORKER_CUH
