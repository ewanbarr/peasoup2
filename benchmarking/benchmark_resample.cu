#include <sstream>
#include <iostream>
#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/timeseries.cuh"
#include "transforms/resampler.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"

using namespace peasoup;

template<System system>
void resample_benchmark(size_t size, float accel, int nits=100)
{
    type::TimeSeries<system,float> input;
    type::TimeSeries<system,float> output;
    transform::TimeDomainResampler<system,float> resamp(input,output);
    utils::Timer clock;
    input.data.resize(size);
    input.metadata.tsamp = 0.000064;
    resamp.prepare();
    resamp.set_accel(accel);
    clock.start();
    for (int ii=0;ii<nits;ii++)
	resamp.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("TimeDomainResampler: ",size," pt: ",accel," m/s/s: ",clock.elapsed()/nits," ms avg\n");
    std::cout << size << "\t" << accel << "\t" <<clock.elapsed()/nits << "\n";
};

int main( int argc, char *argv[] )
{
    if (argc == 2)
	{
	    size_t size = (size_t) atoi(argv[1]);
	    resample_benchmark<DEVICE>(size,500.0);
	} 
    else
	{
	    for (float accel=0.0;accel<1000.0;accel+=100.0){
		for (int ii=14;ii<27;ii++)
		    resample_benchmark<DEVICE>(1<<ii,accel);
	    }
	}
    return 0;
}
