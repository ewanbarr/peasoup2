#include "gtest/gtest.h"

#include <thrust/complex.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/frequencyseries.cuh"
#include "transforms/normaliser.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case(size_t size)
{
    typedef thrust::complex<T> complex;
    int ii;
    //ALL HAIL THE RNG!
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<T> dist(0.0f, 1.0f);
    
    //Build input
    type::FrequencySeries<HOST,complex> input;
    input.metadata.binwidth = 1/600.;
    input.metadata.dm = 235.3;
    input.metadata.acc = 0.33;
    input.data.resize(size);
    
    type::FrequencySeries<HOST,T> baseline;
    baseline.metadata = input.metadata;
    baseline.data.resize(size);
    
    //create a random complex vector 
    //with monotonically increasing variance
    for (ii=0;ii<size;ii++)	{
	input.data[ii] = complex(dist(rng),dist(rng));
	baseline.data[ii] = 5.0;
    }
    
    type::FrequencySeries<system,complex> x=input;
    type::FrequencySeries<system,complex> y;
    type::FrequencySeries<system,T> z=baseline;
    
    transform::Normaliser<system,T> normaliser(x,y,z);
    normaliser.prepare();
    ASSERT_EQ(x.data.size(),size);
    ASSERT_EQ(x.data.size(),y.data.size());
    ASSERT_EQ(x.data.size(),z.data.size());
    ASSERT_EQ(x.metadata.binwidth,y.metadata.binwidth);
    ASSERT_EQ(x.metadata.dm,y.metadata.dm);
    ASSERT_EQ(x.metadata.acc,y.metadata.acc);
	
    normaliser.normalise();
    
    //Create output
    type::FrequencySeries<HOST,complex> output = y;
    
    for (ii=0;ii<size;ii++){
	complex a = sqrt(LN4/baseline.data[ii])*input.data[ii];
	complex b = output.data[ii];
	ASSERT_FLOAT_EQ(a.real(),b.real());
	ASSERT_FLOAT_EQ(a.imag(),b.imag());
    }
}

TEST(NormaliserTest, HostNormalise)
{ test_case<HOST,float>(1<<18); }

TEST(NormaliserTest, DeviceNormalise)
{ test_case<DEVICE,float>(1<<18); }
