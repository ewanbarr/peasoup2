#include <cmath>
#include "gtest/gtest.h"
#include <thrust/sequence.h>
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/harmonicsum.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case(size_t size, unsigned nharms)
{
    size_t ii,jj,kk;
    //Build input
    
    type::FrequencySeries<HOST,T> in;
    in.data.resize(size);
    for (ii=0;ii<size;ii++)
	in.data[ii] = ii/10000.0;
    in.metadata.binwidth = 0.001;
    in.metadata.dm = 235.3;
    in.metadata.acc = 0.33;
    
    
    type::FrequencySeries<system,T> x = in;

    //Create output
    type::HarmonicSeries<system,T> y;
    
    //Instantiate transform
    transform::HarmonicSum<system,T> summer(x,y,nharms);
    summer.prepare();
    for (ii=0;ii<nharms;ii++)
	ASSERT_EQ(x.metadata.binwidth/(1<<(ii+1)), y.metadata.binwidths[ii]);
    ASSERT_EQ(x.metadata.dm,y.metadata.dm);
    ASSERT_EQ(x.metadata.acc,y.metadata.acc);
    ASSERT_EQ(x.data.size()*nharms,y.data.size());
    
    summer.execute();
    
    //check outputs
    type::HarmonicSeries<HOST,T> out=y;
    
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    
    for (ii=0;ii<nharms;ii++){
	for (jj=0;jj<size;jj++){
	    T val = 0;
	    float fjj = (float) jj;
	    for (kk=1;kk<(1<<(ii+1))+1;kk++)
		val += in.data[kk*fjj/(1<<(ii+1))+0.5];
	    ASSERT_NEAR(out.data[ii*size+jj],val,0.01);
	}
    }
}

TEST(HarmonicSumTest, HostSum)
{ test_case<HOST,float>(1<<18,5); }

TEST(HarmonicSumTest, DeviceSum)
{ test_case<DEVICE,float>(1<<18,5); }

