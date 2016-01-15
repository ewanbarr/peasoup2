#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/baselinefinder.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case(size_t size)
{
    int ii;
    type::FrequencySeries<HOST,T> hinput;
    hinput.metadata.dm = 0;
    hinput.metadata.acc = 0;
    hinput.metadata.binwidth = 1/600.0;
    hinput.data.resize(size);
    
    for (ii=0;ii<size;ii++)
	hinput.data[ii] = ii;
    
    type::FrequencySeries<system,T> input = hinput;
    type::FrequencySeries<system,T> baseline;
    transform::BaselineFinder<system,T> baselinefinder(input,baseline,200.0);
    baselinefinder.prepare();
    
    const std::vector<typename SystemVector<system,T>::vector_type>& medians = baselinefinder.get_medians();
    const std::vector< size_t >& boundaries = baselinefinder.get_boundaries();
    
    ASSERT_EQ(input.data.size(),size);
    ASSERT_EQ(input.data.size(),baseline.data.size());
    int width = 5;
    for (auto ar: medians){
	ASSERT_EQ(ar.size(),size/width);
	width*=5;
    }
    baselinefinder.find_baseline();

    type::FrequencySeries<HOST,T> out = baseline;
    for (ii=0; ii<out.data.size(); ii++){
	ASSERT_NEAR(out.data[ii],hinput.data[ii],0.1);
    }
}

TEST(BaselineFinderTest, HostSimple)
{
    test_case<HOST, float>(1<<16);
}

TEST(BaselineFinderTest, DeviceSimple)
{
    test_case<DEVICE, float>(1<<16);
}
