#include <vector>

#include "gtest/gtest.h"

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

#include "tvgs/timeseries_generator.cuh"
#include "misc/system.cuh"
#include "data_types/candidates.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/peakfinder.cuh"
#include "transforms/harmonicsum.cuh"
#include "transforms/distillers.cuh"
#include "utils/utils.cuh"
#include "pipelines/fft_based/accelsearcher.cuh"
#include "pipelines/fft_based/preprocessor.cuh"

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
    std::vector<type::Detection> final_cands;
    transform::PeakFinder<system,T> finder(in,in_harms,dets,10.0);
    finder.prepare();
    finder.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    
    transform::HarmonicDistiller still(0.001,5);
    still.distill(dets,final_cands);
    ASSERT_NEAR(final_cands[0].freq,16*16*16*powers.metadata.binwidth,0.001);
    ASSERT_EQ(final_cands.size(),1);
    for (auto cand:final_cands)
	{
	    printf("F: %f  P: %f  S: %f  H: %d  C: %d\n",
		   cand.freq,cand.power,cand.sigma,cand.nh,cand.associated.size());
	}
}

TEST(DistillerTest, HostPresum)
{ test_case<HOST,float>(); }

TEST(DistillerTest, DevicePresum)
{ test_case<DEVICE,float>(); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

