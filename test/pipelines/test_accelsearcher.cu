#include <vector>
#include <utility>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/accelsearcher.cuh"
#include "data_types/timeseries.cuh"
#include "utils/utils.cuh"
#include "tvgs/timeseries_generator.cuh"

using namespace peasoup;

template <System system>
void test_case()
{
    pipeline::AccelSearchArgs args;
    args.acc_list.push_back(1.0);
    args.acc_list.push_back(2.0);
    args.acc_list.push_back(3.0);
    args.minsigma = 6.0;
    args.nharm = 4;
    type::TimeSeries<HOST,float> hinput;
    hinput.data.resize(1<<21);
    hinput.metadata.tsamp = 0.000064;
    generator::make_noise(hinput,0.0f,1.0f);
    generator::add_tone(hinput,123.0f);
    type::TimeSeries<system,float> input = hinput;
    std::vector<type::Detection> dets;
    pipeline::AccelSearch<system> accsearch(input,dets,args);
    accsearch.prepare();
    accsearch.run();

    for (auto& i:dets){
	printf("nh: %d    freq: %f     pow: %f\n",i.nh,i.freq,i.power);
    }

}

TEST(PreprocessorTest, HostTest)
{ test_case<HOST>(); }


TEST(PreprocessorTest, DeviceTest)
{ test_case<DEVICE>(); }

