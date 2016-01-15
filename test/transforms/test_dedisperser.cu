#include <vector>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/timefrequency.cuh"
#include "data_types/dispersiontime.cuh"
#include "transforms/dedisperser.cuh"

using namespace peasoup;

void fill_input(type::TimeFrequencyBits<HOST>& input, size_t nsamps)
{
    input.metadata.tsamp = 0.000064;
    input.metadata.nchans = 1024;
    input.metadata.foff = -0.390;
    input.metadata.fch1 = 1510.0;
    uint8_t bits_per_byte = 8/input.nbits;
    input.data.assign(input.metadata.nchans*nsamps/bits_per_byte,0);
    ASSERT_EQ(input.get_nsamps(),nsamps);
}

void test_case(size_t nsamps, uint8_t nbits)
{
    //Build input
    type::TimeFrequencyBits<HOST> input(nbits);
    fill_input(input,nsamps);
    type::DispersionTime<HOST,uint8_t> output;
    transform::Dedisperser dedisp(input,output,1);
    std::vector<float> dm_list;
    dm_list.push_back(0);
    dedisp.set_dmlist(dm_list);
    dedisp.prepare();
    dedisp.execute();
    ASSERT_EQ(output.data.size(),nsamps);
}

void test_case2(size_t nsamps, uint8_t nbits)
{
    //Build input
    type::TimeFrequencyBits<HOST> input(nbits);
    fill_input(input,nsamps);
    type::DispersionTime<HOST,uint8_t> output;
    transform::Dedisperser dedisp(input,output,1);
    dedisp.gen_dmlist(0.0,10.0,40.0,1.05);
    const std::vector<float>& dm_list = dedisp.get_dmlist();
    for (int ii=0;ii<dm_list.size()-1;ii++){
	printf("Generated DM (%d):  %.3f\n",ii,dm_list[ii]);
	ASSERT_TRUE(dm_list[ii]<10.0);
    }
    dedisp.prepare();
    dedisp.execute();
}

TEST(DedisperserTest, ZeroDM)
{ test_case(1<<16,2); }

TEST(DedisperserTest, DMListGen)
{ test_case2(1<<16,2); }


