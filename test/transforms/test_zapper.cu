#include <vector>
#include <utility>

#include "gtest/gtest.h"

#include "thrust/complex.h"

#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/zapper.cuh"

using namespace peasoup;


template <System system, typename T>
void test_case(size_t size)
{
    typedef std::pair<float,float> bird;
    typedef thrust::complex<T> complex;
    
    type::FrequencySeries<system,complex> input;
    float df = 0.01;
    input.metadata.binwidth = df;
    input.data.resize(size,complex(1.0,1.0));
    
    std::vector<bird> birdies;
    birdies.push_back(bird(0.3,0.05));
    birdies.push_back(bird(12.3,0.15));
    birdies.push_back(bird(33.3,1.0));
    birdies.push_back(bird(333.3,2.0));
    birdies.push_back(bird(999.3,2.0)); //<---outside of range

    std::vector<bool> mask(size,false);
    
    for (bird i: birdies) {
	unsigned lower,upper;
	float freq = std::get<0>(i);
	float width = std::get<1>(i);
	int bin = (freq/df + 0.5);
	int bins = (width/df);
	lower = std::max(bin-bins,0);
	upper = std::min(bin+bins,(int)size);
	for (int ii=lower;ii<upper;ii++)
	    mask[ii] = true;
    }
    
    transform::Zapper<system,T> zapper(input,birdies);
    zapper.prepare();
    zapper.execute();
    type::FrequencySeries<HOST,complex> output = input;

    for (int jj=0;jj<size;jj++){
	if (mask[jj])
	    ASSERT_EQ(output.data[jj],complex(0.0,0.0));
	else
	    ASSERT_EQ(output.data[jj],complex(1.0,1.0));
    }
    
}

TEST(ZapperTest, HostTest)
{ test_case<HOST,float>(1<<16); }

TEST(ZapperTest, DeviceTest)
{ test_case<DEVICE,float>(1<<16); }


