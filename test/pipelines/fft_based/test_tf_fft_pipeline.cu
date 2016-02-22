#include "gtest/gtest.h"

#include "misc/system.cuh"
#include "pipelines/fft_based/tf_fft_pipeline.cuh"
#include "pipelines/args.hpp"
#include "data_types/timefrequency.cuh"
#include "utils/utils.cuh"
#include "pipelines/cmdline.cuh"

using namespace peasoup;

int my_argc;
char** my_argv;

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
    cmdline::read_cmdline_options(opts,my_argc,my_argv);
    if (opts.tf_fft_args.accelsearch.nfft == 0)
	opts.tf_fft_args.accelsearch.nfft = size;
    //opts.tf_fft_args.accelsearch.user_acc_list.push_back(0.0);
    type::TimeFrequencyBits<HOST> input(2);
    fill_input(input,size);
    pipeline::TimeFrequencyFFTPipeline pipeline(input,opts.tf_fft_args);
    pipeline.prepare();
    pipeline.run();
}

TEST(TimeFrequencyFFTPipelineTest, TestExecute)
{ test_case(1<<18); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    my_argc = argc;
    my_argv = argv;
    return RUN_ALL_TESTS();
}
