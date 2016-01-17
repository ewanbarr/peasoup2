#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/timeseries.cuh"
#include "tvgs/timeseries_generator.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

TEST(TvgTest, WriteTvg)
{
    type::TimeSeries<HOST,float> x;
    x.metadata.tsamp = 0.000064;
    x.data.resize(1<<21);
    generator::make_noise(x);
    generator::add_pulse_train(x,10.0,10.0,0.1);
    //utils::write_vector<decltype(x.data)>(x.data,"test_vector.bin");
}
