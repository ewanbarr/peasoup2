#include <vector>

#include "gtest/gtest.h"

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

#include "misc/system.cuh"
#include "data_types/candidates.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/peakfinder.cuh"
#include "transforms/harmonicsum.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case()
{
    int ii;
    //Build input
    type::FrequencySeries<HOST,T> powers;
    powers.metadata.binwidth = 1/600.0;
    powers.metadata.dm = 235.3;
    powers.metadata.acc = 0.33;
    powers.data.resize(1<<22,0);
    for (ii = 1; ii< 17; ii++)
	powers.data[16*16*16*ii] = 1000.0;

    type::HarmonicSeries<HOST,T> harms;
    transform::HarmonicSum<HOST,T> sum(powers,harms,4);
    sum.prepare();
    sum.execute();
    type::FrequencySeries<system,T> in = powers;
    type::HarmonicSeries<system,T> in_harms = harms;
    std::vector<type::Detection> dets;
    transform::PeakFinder<system,T> finder(in,in_harms,dets,10.0);
    finder.prepare();
    finder.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    std::vector<float>& thresholds = finder.get_thresholds();

    float max_power,max_freq;
    max_power = max_freq = 0.0;
    for (auto& i: dets){
	if (i.power > max_power){
	    max_power = i.power;
	    max_freq = i.freq;
	}
    }
    ASSERT_NEAR(max_freq,16*16*16*powers.metadata.binwidth,0.0001);
    ASSERT_NEAR(max_power,16*1000.0,0.0001);
}

TEST(PeakFinderTest, HostFinder)
{ test_case<HOST,float>(); }

TEST(PeakFinderTest, DeviceFinder)
{ test_case<DEVICE,float>(); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

