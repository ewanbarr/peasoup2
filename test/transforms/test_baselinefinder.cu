#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/timeseries.cuh"
#include "transforms/baselinefinder.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

template <System system, typename T>
void fd_test_case(size_t size)
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
    transform::FDBaselineFinder<system,T> baselinefinder(input,baseline,200.0);
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
    baselinefinder.execute();
    type::FrequencySeries<HOST,T> out = baseline;
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    for (ii=0; ii<out.data.size(); ii++){
	ASSERT_NEAR(out.data[ii],hinput.data[ii],0.1);
    }
}

template <System system, typename T>
void td_test_case(size_t size)
{
    int ii;
    type::TimeSeries<HOST,T> hinput;
    hinput.metadata.dm = 0;
    hinput.metadata.acc = 0;
    hinput.metadata.tsamp = 0.000064;
    hinput.data.resize(size);

    for (ii=0;ii<size;ii++)
        hinput.data[ii] = ii;

    type::TimeSeries<system,T> input = hinput;
    type::TimeSeries<system,T> baseline;
    transform::TDBaselineFinder<system,T> baselinefinder(input,baseline,2.0);
    baselinefinder.prepare();
    
    const std::vector<typename SystemVector<system,T>::vector_type>& medians = baselinefinder.get_medians();

    ASSERT_EQ(input.data.size(),size);
    ASSERT_EQ(input.data.size(),baseline.data.size());
    int width = 5;
    for (auto ar: medians){
        ASSERT_EQ(ar.size(),size/width);
        width*=5;
    }
    baselinefinder.execute();
    type::TimeSeries<HOST,T> out = baseline;
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    for (ii=0; ii<out.data.size(); ii++){
        ASSERT_NEAR(out.data[ii],hinput.data[ii],0.1);
    }
}


TEST(FDBaselineFinderTest, HostSimple)
{
    fd_test_case<HOST, float>(1<<16);
}

TEST(FDBaselineFinderTest, DeviceSimple)
{
    fd_test_case<DEVICE, float>(1<<16);
}

TEST(TDBaselineFinderTest, HostSimple)
{
    td_test_case<HOST, float>(1<<16);
}

TEST(TDBaselineFinderTest, DeviceSimple)
{
    td_test_case<DEVICE, float>(1<<16);
}
