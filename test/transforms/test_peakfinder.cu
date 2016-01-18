#include <vector>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/candidates.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/peakfinder.cuh"
#include "transforms/harmonicsum.cuh"

using namespace peasoup;

template <System system, typename T>
void test_case()
{
    //Build input
    type::FrequencySeries<HOST,T> x;
    x.metadata.binwidth = 1/600.0;
    x.metadata.dm = 235.3;
    x.metadata.acc = 0.33;
    x.data.resize(4096);
    for (int ii=0;ii<4096;ii++)
	x.data[ii] = 0;
    
    x.data[1111] = 5.0;
    x.data[1334] = 20.0;
    x.data[11] = 30.0;
    x.data[4095] = 55.0;

    type::HarmonicSeries<HOST,T> harms;
    transform::HarmonicSum<HOST,T> sum(x,harms,4);
    sum.prepare();
    
    harms.data[4096*0 + 1111] = 23;
    harms.data[4096*1 + 1111] = 24;
    harms.data[4096*2 + 1111] = 25;
    harms.data[4096*3 + 1111] = 26;

    
    type::FrequencySeries<system,T> in = x;
    type::HarmonicSeries<system,T> in_harms = harms;
    std::vector<type::Detection> dets;
    
    transform::PeakFinder<system,T> finder(in,in_harms,dets,3.0);
    finder.prepare();
    finder.execute();
    
    std::vector<float>& thresholds = finder.get_thresholds();
    for (float x: thresholds)
	printf("Thresh:    %f\n",x);

    for (auto& i: dets){
	printf("%f    %f\n",i.freq/x.metadata.binwidth,i.power);
    }
    
}

TEST(PeakFinderTest, HostFinder)
{ test_case<HOST,float>(); }
