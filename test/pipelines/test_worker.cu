#include "gtest/gtest.h"

#include <vector>
#include <chrono>
#include <thread>

#include "misc/system.cuh"
#include "pipelines/dmtrialqueue.cuh"
#include "pipelines/worker.cuh"
#include "pipelines/fft_based/accelsearch_worker.cuh"
#include "pipelines/args.hpp"
#include "data_types/dispersiontime.cuh"

using namespace peasoup;

typedef std::pair<float,float> bird;

template <System system>
void test_case(size_t size, int ndms, int nthreads)
{
    typedef typename type::DispersionTime<system,uint8_t> trial_type;
    typedef typename pipeline::DMTrialQueue<trial_type> queue_type;
    typedef typename pipeline::AccelSearchWorker<system, queue_type > worker_type;
    int ii;
    trial_type trials;
    trials.data.resize(size*ndms);
    for (ii=0; ii<ndms;ii++)
        trials.metadata.dms.push_back((float) ii);
    trials.metadata.tsamp = 0.000064;
    
    queue_type queue(trials);
    pipeline::AccelSearchArgs args;
    args.birdies.clear();
    for (ii=0;ii<10;ii++)
        args.acc_list.push_back((float)ii);
    args.birdies.push_back(bird(123.0,0.2));
    args.minsigma = 6.0;
    args.nharm = 4;
    args.nfft = size;
    pipeline::WorkerPool<worker_type,queue_type,pipeline::AccelSearchArgs> pool(queue,args,nthreads);
    pool.prepare();
    pool.run();
    pool.join();
}


TEST(AccelSearchWorkerTest, TestSingleThreadHost)
{ test_case<HOST>(1<<18, 4, 1); }

TEST(AccelSearchWorkerTest, TestSingleThreadDevice)
{ test_case<DEVICE>(1<<18, 4, 1);}

TEST(AccelSearchWorkerTest, TestMultiThreadHost)
{ test_case<HOST>(1<<18, 4, 4);}

TEST(AccelSearchWorkerTest, TestMultiThreadDevice)
{ test_case<DEVICE>(1<<18, 4, 4);}
