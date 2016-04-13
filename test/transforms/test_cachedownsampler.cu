#include "gtest/gtest.h"

#include <stdexcept>

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include "cuda.h"
#include "data_types/timeseries.cuh"
#include "transforms/downsampler.cuh"
#include "utils/logging.hpp"

using namespace peasoup;

template <System system>
void test_case(size_t in_size, unsigned factor)
{
    typedef transform::CachedDownsampler<system,float> cache;
    
    int ii;
    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    
    type::TimeSeries<HOST,float> hin;
    hin.data.resize(in_size);
    hin.metadata.tsamp = 0.000064;
    hin.metadata.dm = 0;
    hin.metadata.acc = 0;
    
    for (ii=0;ii<in_size;ii++){
	hin.data[ii] = ii;
    }
    
    type::TimeSeries<system,float> din = hin;
    cache downsampler(&din);

    for (int jj=1;jj<123;jj++){
	unsigned nearest = downsampler.closest_factor(jj);
	printf("Nearest factor to %d is %d\n",jj,nearest);
	cache* node = downsampler.downsample(nearest);
	node->data->metadata.display();
	if (jj>1)
	    for (int kk=0;kk<node->data->data.size();kk++){
		ASSERT_FLOAT_EQ(node->data->data[kk],(kk+1)*nearest-nearest/2.0-0.5);
	    }
    }


    for (ii=0;ii<in_size;ii++){
        hin.data[ii] = 1.0;
    }
    type::TimeSeries<system,float> din2 = hin;
    downsampler.set_data(&din2);

    for (int jj=1;jj<123;jj++){
        unsigned nearest = downsampler.closest_factor(jj);
        printf("Nearest factor to %d is %d\n",jj,nearest);
        cache* node = downsampler.downsample(nearest);
        node->data->metadata.display();
        if (jj>1)
            for (int kk=0;kk<node->data->data.size();kk++){
                ASSERT_FLOAT_EQ(node->data->data[kk],1.0);
            }
    }

}

TEST(DownsampleTest,TestHost)
{ 
    logging::set_default_log_level_from_string("DEBUG");
    test_case<HOST>(1<<12,63); 
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
