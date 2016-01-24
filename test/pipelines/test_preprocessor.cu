#include <vector>
#include <utility>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/preprocessor.cuh"
#include "pipelines/args.hpp"
#include "data_types/timeseries.cuh"
#include "utils/utils.cuh"
#include "tvgs/timeseries_generator.cuh"

using namespace peasoup;

typedef std::pair<float,float> bird;

template <System system>
void test_case()
{
    pipeline::AccelSearchArgs args;
    args.acc_list.push_back(1.0);
    args.birdies.push_back(bird(123.0,0.2));
    type::TimeSeries<HOST,float> hinput;
    hinput.data.resize(1<<21);
    hinput.metadata.tsamp = 0.000064;
    generator::make_noise(hinput,0.0f,1.0f,0.9999f);
    generator::add_tone(hinput,123.0f);
    
    type::TimeSeries<system,float> input = hinput;
    pipeline::Preprocessor<system> preproc(input,input,args);
    preproc.prepare();
    preproc.run();
}

TEST(PreprocessorTest, HostTest)
{ test_case<HOST>(); }


TEST(PreprocessorTest, DeviceTest)
{ test_case<DEVICE>(); }

