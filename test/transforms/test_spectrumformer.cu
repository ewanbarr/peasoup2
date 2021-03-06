#include <cmath>
#include "gtest/gtest.h"
#include <thrust/complex.h>
#include <thrust/extrema.h>
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/spectrumformer.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case(size_t size)
{
    size_t ii,jj;
    //Build input
    
    type::FrequencySeries<HOST,thrust::complex<T> > in;
    in.data.resize(size);
    for (ii=0;ii<size;ii++)
	in.data[ii] = thrust::complex<T>(2.34,-1.33);
    in.metadata.binwidth = 0.001;
    in.metadata.dm = 235.3;
    in.metadata.acc = 0.33;
        
    type::FrequencySeries<system,thrust::complex<T> > x = in;

    //Create output
    type::FrequencySeries<system,T> y;

    //Instantiate transform
    transform::SpectrumFormer<system,T> former(x,y);
    former.prepare();
    ASSERT_EQ(x.metadata.binwidth, y.metadata.binwidth);
    ASSERT_EQ(x.metadata.dm,y.metadata.dm);
    ASSERT_EQ(x.metadata.acc,y.metadata.acc);
    ASSERT_EQ(x.data.size(),y.data.size());
    
    former.form();
    
    //check outputs
    type::FrequencySeries<HOST,T> out = y;

    for (jj=0;jj<size;jj++){
	ASSERT_NEAR(out.data[jj], thrust::abs<T>(in.data[jj]), 0.01);
    }
    
    former.form_nn();

    type::FrequencySeries<HOST,T> out_nn = y;
    
    ASSERT_EQ(out.data[0],0.0);
    for (jj=1;jj<size;jj++){
	T val = thrust::max<T>(thrust::abs<T>(out_nn.data[jj]-out_nn.data[jj-1])*RSQRT2,thrust::abs<T>(out_nn.data[jj]));
	ASSERT_FLOAT_EQ(val,out_nn.data[jj]);
    }
}

TEST(SpectrumFormerTest, HostForm)
{ test_case<HOST,float>(1<<18); }

TEST(SpectrumFormerTest, DeviceForm)
{ test_case<DEVICE,float>(1<<18); }

