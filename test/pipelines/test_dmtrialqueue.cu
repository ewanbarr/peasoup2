#include "gtest/gtest.h"

#include <chrono>
#include <thread>

#include "misc/system.cuh"
#include "pipelines/dmtrialqueue.cuh"
#include "data_types/dispersiontime.cuh"
#include "data_types/timeseries.cuh"

using namespace peasoup;

template <typename QueueType>
void test_consumer(QueueType& queue, size_t size, int id)
{
    typedef typename type::TimeSeries<HOST,float> tim_type;
    tim_type tim;
    while (true){
	tim.data.resize(0);
	if (!queue.pop(tim))
	    break;
	ASSERT_EQ(tim.data.size(),size);
	printf("Thread: %d    Size: %d    DM: %f\n",id,tim.data.size(),tim.metadata.dm);
	std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

TEST(DMTrialQueueTest, TestThreadedAccess)
{
    typedef typename type::DispersionTime<HOST,uint8_t> trial_type;
    size_t nsamples = 1000;
    int ndms = 12;
    trial_type trials;
    trials.data.resize(nsamples*ndms);
    for (int ii=0; ii<ndms;ii++)
	trials.metadata.dms.push_back((float) ii);
    pipeline::DMTrialQueue<trial_type> queue(trials);
    std::thread a (test_consumer< typename pipeline::DMTrialQueue<trial_type> >, std::ref(queue), nsamples, 1);
    std::thread b (test_consumer< typename pipeline::DMTrialQueue<trial_type> >, std::ref(queue), nsamples, 2);
    a.join();
    b.join();
}

