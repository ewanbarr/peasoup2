#include <algorithm>
#include <cstdlib>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/types.cuh"

using namespace peasoup;

template <System A, System B>
void test_case(size_t size, unsigned nchans)
{
    type::TimeFrequency<A,float> x;
    x.data.resize(size);
    x.metadata.nchans = nchans;
    std::generate(x.data.begin(),x.data.end(),std::rand);
    type::TimeFrequency<B,float> y = x;
    ASSERT_EQ(x.data.size(),y.data.size());
    for (int ii=0;ii<size;ii++)
        ASSERT_EQ(x.data[ii],y.data[ii]);
    ASSERT_EQ(x.get_nsamps(),size/nchans);
}

TEST(TimeFrequencyTest, ImplicitHtoD)
{ test_case<HOST,DEVICE>(1<<12,1<<7); }  

TEST(TimeFrequencyTest, ImplicitDtoH)
{ test_case<DEVICE,HOST>(1<<12,1<<6); }

TEST(TimeFrequencyTest, ImplicitHtoH)
{ test_case<HOST,HOST>(1<<12,1<<5); }

TEST(TimeFrequencyTest, ImplicitDtoD)
{ test_case<DEVICE,DEVICE>(1<<12,1<<4); }

TEST(TimeFrequencyBits, TestGetNsamps)
{ 
    unsigned nbits = 2;
    unsigned bits_per_byte = 8/nbits;
    unsigned nchans = 32;
    unsigned nsamps = 1024;
    unsigned size = nsamps*nchans/bits_per_byte;
    type::TimeFrequencyBits<HOST> x(2);
    x.data.resize(size);
    x.metadata.nchans = nchans;
    ASSERT_EQ(x.get_nsamps(),nsamps);
}
