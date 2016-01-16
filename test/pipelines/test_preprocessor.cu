#include <vector>
#include <utility>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/preprocessor.cuh"
#include "data_types/timeseries.cuh"

using namespace peasoup;

template <System system>
void test_case()
{
    PeasoupArgs args;
    args.acc_list.push_back(100.0);
    type::TimeSeries<system,float> input;
    input.data.resize(1<<21);
    input.metadata.tsamp = 0.000064;
    input.metadata.acc = 0;
    input.metadata.dm = 0;
    pipeline::Preprocessor<system> preproc(input,input,args);
    preproc.prepare();
    preproc.run();
}

TEST(PreprocessorTest, HostTest)
{ test_case<HOST>(); }

TEST(PreprocessorTest, DeviceTest)
{ test_case<DEVICE>(); }
