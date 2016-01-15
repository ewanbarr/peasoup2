#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/timeseries.cuh"
#include "transforms/resampler.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case(size_t size, float accel)
{
    //Build input
    type::TimeSeries<system,T> x;
    x.metadata.tsamp = 0.000064;
    x.metadata.dm = 235.3;
    x.metadata.acc = 0.33;
    x.data.resize(size);

    //Create output
    type::TimeSeries<system,T> y;
    
    //Instantiate transform
    transform::TimeDomainResampler<system,T> resamp(x,y);
    resamp.prepare();
    ASSERT_EQ(x.metadata.tsamp, y.metadata.tsamp);
    ASSERT_EQ(x.metadata.dm,y.metadata.dm);
    ASSERT_EQ(x.metadata.acc,y.metadata.acc);
    ASSERT_EQ(x.data.size(),y.data.size());

    resamp.resample(accel);
    
    //check outputs
    type::TimeSeries<HOST,T> in = x;
    type::TimeSeries<HOST,T> out = y;
    double accel_fact = ((accel * in.metadata.tsamp) / (2 * SPEED_OF_LIGHT));
    double dsize = (double) size;
    for (size_t ii=0;ii<size;ii++)
	ASSERT_TRUE(out.data[ii] == in.data[(size_t)(ii + ii*accel_fact*(ii-dsize))]);
}

TEST(ResamplerTest, HostResampleHighAcc)
{ test_case<HOST,float>(1<<20,1000.0); }

TEST(ResamplerTest, HostResampleLowAcc)
{ test_case<HOST,float>(1<<20,0.4); }

TEST(ResamplerTest, DeviceResampleHighAcc)
{ test_case<DEVICE,float>(1<<20,1000.0); }

TEST(ResamplerTest, DeviceResampleLowAcc)
{ test_case<DEVICE,float>(1<<20,0.4); }

