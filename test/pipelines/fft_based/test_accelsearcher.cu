#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <ctime>
#include <tclap/CmdLine.h>

#include "gtest/gtest.h"
#include "thrust/complex.h"

#include "misc/system.cuh"
#include "pipelines/fft_based/accelsearcher.cuh"
#include "pipelines/fft_based/preprocessor.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/candidates.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/distillers.cuh"
#include "tvgs/timeseries_generator.cuh"
#include "utils/utils.cuh"

using namespace peasoup;


namespace test_args {
    int my_argc;
    char** my_argv;
    
    struct AccelsearchTestArgs {
	int nharm;
	float minsigma;
	float freq;
	float dc;
	float snr;
	float acc_start;
	float acc_end;
	float acc_step;
	bool host;
    };
    
    bool read_cmdline(AccelsearchTestArgs& args)
    {
	try {
	    TCLAP::CmdLine cmd("test_accelsearcher (test accelsearch pipeline)", ' ', " ");

	    TCLAP::ValueArg<float> arg_acc_start("", "acc_start",
						 "First acceleration to resample to",
						 false, 0.0, "float", cmd);
	    
	    TCLAP::ValueArg<float> arg_acc_end("", "acc_end",
					       "Last acceleration to resample to",
					       false, 0.0, "float", cmd);
	    
	    TCLAP::ValueArg<float> arg_acc_step("", "acc_step",
						"Acceleration step",
						false, 1.0, "float",cmd);
	    
	    TCLAP::ValueArg<int> arg_nharm("", "nharm",
					   "Number of harmonics to sum",
					   false, 5, "int",cmd);

	    TCLAP::ValueArg<float> arg_minsigma("", "minsigma",
                                                "minimum sigma for search",
                                                false, 5.0, "float",cmd);
	    
	    TCLAP::ValueArg<float> arg_freq("", "freq",
					    "frequency of injected signal",
					    false, 125.000000000, "float",cmd);
	    
	    TCLAP::ValueArg<float> arg_dc("", "dc",
					  "duty cycle of injected signal",
					  false, 0.1, "float",cmd);
	    
	    TCLAP::ValueArg<float> arg_snr("", "snr",
					   "snr cycle of injected signal",
					   false, 15.0, "float",cmd);
	    
	    
	    TCLAP::SwitchArg arg_host("", "host", "run on host", cmd);

	    cmd.parse(my_argc, my_argv);
	    args.acc_start         = arg_acc_start.getValue();
	    args.acc_end           = arg_acc_end.getValue();
	    args.acc_step          = arg_acc_step.getValue();
	    args.nharm             = arg_nharm.getValue();
	    args.minsigma          = arg_minsigma.getValue();
	    args.freq              = arg_freq.getValue();
	    args.dc                = arg_dc.getValue();
	    args.snr               = arg_snr.getValue();
	    args.host              = arg_host.getValue();
	    
	}catch (TCLAP::ArgException &e) {
	    std::cerr << "Error: " << e.error() << " for arg " << e.argId()
		      << std::endl;
	    return false;
	}
	return true;
    }
    
}

template <System system>
void test_case(test_args::AccelsearchTestArgs& in_args)
{
    size_t size = 1<<23;
    float tsamp = 0.000064;
    pipeline::AccelSearchArgs args;
    for (float accel=in_args.acc_start;
	 accel<=in_args.acc_end;
	 accel+=in_args.acc_step)
    	args.user_acc_list.push_back(accel);
    args.minsigma = in_args.minsigma;
    args.nharm = in_args.nharm;
    args.nfft = size;
    type::TimeSeries<HOST,float> hinput;
    hinput.data.resize(size);
    hinput.metadata.tsamp = tsamp;
    generator::make_noise(hinput,0.0f,1.0f);
    generator::add_pulse_train(hinput,in_args.freq,in_args.snr,in_args.dc);
    type::TimeSeries<system,float> input = hinput;
    std::vector<type::Detection> dets;
    pipeline::Preprocessor<system> preproc(input,input,args);
    pipeline::AccelSearch<system> accsearch(input,dets,args);
    preproc.prepare();
    accsearch.prepare();
    preproc.run();
    accsearch.run();
    type::FrequencySeries<HOST,thrust::complex<float> > fourier = accsearch.fourier;
    type::FrequencySeries<HOST,float> spec = accsearch.spectrum;
    type::HarmonicSeries<HOST,float> hsum = accsearch.harmonics;
    for (auto& i:dets){
	printf("nh: %d    freq: %f     pow: %f    sigma: %f\n",i.nh,i.freq,i.power,i.sigma);
    }
    transform::HarmonicDistiller still(0.001,in_args.nharm);

}

TEST(AccelsearchTest, Test)
{ 
    test_args::AccelsearchTestArgs args;
    test_args::read_cmdline(args);
    if (args.host)
	test_case<HOST>(args); 
    else
	test_case<DEVICE>(args);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    test_args::my_argc = argc;
    test_args::my_argv = argv;
    return RUN_ALL_TESTS();
}
