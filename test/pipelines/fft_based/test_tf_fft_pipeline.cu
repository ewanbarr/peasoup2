#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/fft_based/tf_fft_pipeline.cuh"
#include "pipelines/args.hpp"
#include "data_types/timefrequency.cuh"
#include "utils/utils.cuh"

using namespace peasoup;

typedef std::pair<float,float> bird;

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

void test_case(size_t size)
{
    pipeline::Options opts;
    opts.minsigma = 6.0;
    opts.nharm = 4;
    opts.ngpus = 1;
    opts.nthreads = 3;
    opts.nfft = size;
    type::TimeFrequencyBits<HOST> input(2);
    fill_input(input,size);

    pipeline::TimeFrequencyFFTPipeline pipeline(input,opts);
    pipeline.prepare();
    pipeline.run();
}

TEST(TimeFrequencyFFTPipelineTest, TestExecute)
{ test_case(1<<23); }


