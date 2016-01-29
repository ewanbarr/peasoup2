#include "gtest/gtest.h"

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include "cuda.h"
#include "data_types/timeseries.cuh"
#include "transforms/pad.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

template <System system>
void test_case(size_t in_size, size_t out_size)
{
    int ii;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    
    type::TimeSeries<HOST,float> hin;
    hin.data.resize(in_size);
    hin.metadata.tsamp = 0.000064;
    hin.metadata.dm = 0;
    hin.metadata.acc = 0;
    
    float mean = 0;
    for (ii=0;ii<in_size;ii++){
	hin.data[ii] = dist(rng);
	mean += hin.data[ii];
    }
    mean /= in_size;
    
    type::TimeSeries<system,float> din = hin;
    transform::Pad<system,float> pad(din,in_size,out_size);
    pad.prepare();
    pad.execute();
    
    type::TimeSeries<HOST,float> hout = din;

    utils::check_cuda_error(__PRETTY_FUNCTION__);

    for (ii=in_size;ii<out_size;ii++)
	{
	    ASSERT_NEAR(hout.data[ii],mean,0.0001);
	}
}

TEST(PadTest,TestHost)
{ test_case<HOST>((1<<12)+123,1<<13); }

TEST(PadTest,TestDevice)
{ test_case<DEVICE>((1<<14)+555,1<<15); }

