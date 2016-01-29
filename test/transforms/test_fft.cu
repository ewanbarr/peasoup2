#include "gtest/gtest.h"

#include "cufft.h"
#include "thirdparty/fftw3.h" //Peasoup fix for fftw3 header

#include <thrust/complex.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/timeseries.cuh"
#include "transforms/fft.cuh"
#include "utils/utils.cuh"
#include "tvgs/timeseries_generator.cuh"

using namespace peasoup;

template <System system>
void test_case(size_t size)
{
    int ii;
    typedef thrust::complex<float> complex;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    
    type::TimeSeries<HOST,float> hin;
    hin.data.resize(size);
    hin.metadata.tsamp = 0.000064;
    hin.metadata.dm = 0;
    hin.metadata.acc = 0;
    
    for (ii=0;ii<size;ii++)
	hin.data[ii] = dist(rng);
    
    type::TimeSeries<system,float> din = hin;
    type::FrequencySeries<system,complex> dout;
    
    transform::RealToComplexFFT<system> r2cfft(din,dout);
    r2cfft.prepare();
    float out_binwidth = 1.0/(din.data.size()*din.metadata.tsamp);
    ASSERT_TRUE(fabs((dout.metadata.binwidth-out_binwidth)/out_binwidth)<0.0001);
    ASSERT_EQ(din.metadata.dm,dout.metadata.dm);
    ASSERT_EQ(din.metadata.acc,dout.metadata.acc);
    ASSERT_EQ(dout.data.size(),din.data.size()/2+1);
    r2cfft.execute();

    transform::ComplexToRealFFT<system> c2rfft(dout,din);
    c2rfft.prepare();
    size_t new_size = 2*(dout.data.size() - 1);
    float out_tsamp = 1.0/(dout.metadata.binwidth * new_size);
    ASSERT_TRUE(fabs((din.metadata.tsamp-out_tsamp)/out_tsamp)<0.0001);
    ASSERT_EQ(din.metadata.dm,dout.metadata.dm);
    ASSERT_EQ(din.metadata.acc,dout.metadata.acc);
    ASSERT_EQ(din.data.size(),new_size);
    c2rfft.execute();
    type::TimeSeries<HOST,float> hout = din;
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    
    for (ii=0;ii<size;ii++)
	{
	    float in = hin.data[ii];
	    float out = hout.data[ii];
	    ASSERT_NEAR(in,out,0.0001);
	}

}

TEST(FFTTest,TestR2CHost)
{ test_case<HOST>(1<<23); }

TEST(FFTTest,TestR2CDevice)
{ test_case<DEVICE>(1<<23); }

