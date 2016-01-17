#include <vector>
#include <utility>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/preprocessor.cuh"
#include "data_types/timeseries.cuh"
#include "utils/utils.cuh"
#include "tvgs/timeseries_generator.cuh"

using namespace peasoup;

typedef std::pair<float,float> bird;

template <System system>
void test_case()
{
    PeasoupArgs args;
    args.acc_list.push_back(1.0);
    args.birdies.push_back(bird(123.0,0.2));
    
    type::TimeSeries<system,float> input;
    input.data.resize(1<<21);
    input.metadata.tsamp = 0.000064;
    generator::make_noise(input,0.0,1.0,0.9999);
    generator::add_tone(input,123.0);
    pipeline::Preprocessor<system> preproc(input,input,args);
    preproc.prepare();
    preproc.run();
}

TEST(PreprocessorTest, HostTest)
{ test_case<HOST>(); }


TEST(PreprocessorTest, DeviceTest)
{ test_case<DEVICE>(); }

