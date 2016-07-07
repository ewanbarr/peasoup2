#include "gtest/gtest.h"

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include "cuda.h"
#include "data_types/timeseries.cuh"
#include "transforms/clip.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

template <System system>
void test_case(size_t in_size)
{
    int ii;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    
    type::TimeSeries<HOST,float> hin;
    hin.data.resize(in_size);
    hin.metadata.tsamp = 0.000064;
    hin.metadata.dm = 0;
    hin.metadata.acc = 0;
    
    float thresh = 1.0;

    type::TimeSeries<HOST,float> idxs;

    for (ii=0;ii<in_size;ii++){
	hin.data[ii] = dist(rng);
	if (hin.data[ii] > thresh)
	    idxs.data.push_back(ii);
    }
    
    type::TimeSeries<system,float> din = hin;
    transform::Clip<system,float> pad(din,din,thresh);
    pad.prepare();
    pad.execute();
    
    type::TimeSeries<HOST,float> hout = din;

    utils::check_cuda_error(__PRETTY_FUNCTION__);

    for (auto& idx: idxs.data)
	ASSERT_TRUE(hout.data[idx]<=thresh);

    for (auto& val: hout.data)
	ASSERT_TRUE(val<=thresh);
}

TEST(ClipTest,TestHost)
{ test_case<HOST>((1<<12)); }

TEST(ClipTest,TestDevice)
{ test_case<DEVICE>((1<<14)); }

