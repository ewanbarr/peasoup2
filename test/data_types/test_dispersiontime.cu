#include <algorithm>
#include <cstdlib>

#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "data_types/types.cuh"

#define SIZE 999

using namespace peasoup;

template <System A, System B>
void test_case(size_t size)
{
    type::DispersionTime<A,float> x;
    x.data.resize(SIZE);
    std::generate(x.data.begin(),x.data.end(),std::rand);
    type::DispersionTime<B,float> y = x;
    ASSERT_EQ(x.data.size(),y.data.size());
    for (int ii=0;ii<SIZE;ii++)
        ASSERT_EQ(x.data[ii],y.data[ii]);
}

TEST(DispersionTimeTest, ImplicitHtoD)
{ test_case<HOST,DEVICE>(SIZE); }  

TEST(DispersionTimeTest, ImplicitDtoH)
{ test_case<DEVICE,HOST>(SIZE); }

TEST(DispersionTimeTest, ImplicitHtoH)
{ test_case<HOST,HOST>(SIZE); }

TEST(DispersionTimeTest, ImplicitDtoD)
{ test_case<DEVICE,DEVICE>(SIZE); }


