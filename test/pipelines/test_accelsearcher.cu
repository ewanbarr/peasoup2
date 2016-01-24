#include <vector>
#include <utility>

#include "gtest/gtest.h"
#include "thrust/complex.h"

#include "misc/system.cuh"
#include "pipelines/accelsearcher.cuh"
#include "pipelines/preprocessor.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/candidates.cuh"
#include "data_types/harmonicseries.cuh"
#include "tvgs/timeseries_generator.cuh"

using namespace peasoup;

template <System system>
void test_case()
{
    pipeline::AccelSearchArgs args;
    args.acc_list.push_back(0.0);
    args.minsigma = 6.0;
    args.nharm = 4;

    type::TimeSeries<HOST,float> hinput;
    hinput.data.resize(1<<21);
    hinput.metadata.tsamp = 0.000064;

    generator::make_noise(hinput,0.0f,1.0f);
    generator::add_tone(hinput,123.0f,0.01);
    
    type::TimeSeries<system,float> input = hinput;
    std::vector<type::Detection> dets;
    
    pipeline::Preprocessor<system> preproc(input,input,args);
    pipeline::AccelSearch<system> accsearch(input,dets,args);
    preproc.prepare();
    accsearch.prepare();
    preproc.run();
    accsearch.run();
    type::FrequencySeries<HOST,thrust::complex<float> > fourier = accsearch.fourier;
    type::FrequencySeries<HOST,float> spec = accsearch.spectrum;
    type::HarmonicSeries<HOST,float> hsum = accsearch.harmonics;
    for (auto& i:dets){
	printf("nh: %d    freq: %f     pow: %f\n",i.nh,i.freq,i.power);
    }

}

TEST(AccelsearchTest, HostTest)
{ test_case<HOST>(); }


TEST(AccelsearchTest, DeviceTest)
{ test_case<DEVICE>(); }

