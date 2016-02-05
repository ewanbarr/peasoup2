#include <sstream>
#include <iostream>
#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/harmonicsum.cuh"
#include "transforms/peakfinder.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"

using namespace peasoup;

template<System system>
void peakfind_benchmark(size_t size, size_t npeaks, int nits=100)
{
    int step = size/npeaks;
    type::FrequencySeries<HOST,float> hinput;
    hinput.data.resize(size);
    for (int ii=0;ii<size;ii+=step)
	hinput.data[ii] = 110.0;

    type::FrequencySeries<system,float> input=hinput;
    input.metadata.binwidth = 1/600.0;
    input.metadata.dm = 235.3;
    input.metadata.acc = 0.33;
   
    type::HarmonicSeries<system,float> harms;
    transform::HarmonicSum<system,float> sum(input,harms,5);
    sum.prepare();
    
    std::vector<type::Detection> dets;
    transform::PeakFinder<system,float> finder(input,harms,dets,5.0);
    finder.prepare();

    utils::Timer clock;
    clock.start();
    for (int ii=0;ii<nits;ii++)
	finder.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("Peakfinder: ",size," pt: ",clock.elapsed()/nits," ms avg\n");
    std::cout << size << "\t" << npeaks << "\t" <<clock.elapsed()/nits << "\n";
};

int main( int argc, char *argv[] )
{
    if (argc == 3)
	{
	    size_t size = (size_t) atoi(argv[1]);
	    size_t npeaks = (size_t) atoi(argv[2]);
	    peakfind_benchmark<DEVICE>(size,npeaks,1);
	} 
    else
	{
	    for (int ii=14;ii<27;ii++)
		peakfind_benchmark<DEVICE>(1<<ii,(size_t)((1<<ii)/10.0),1);
	}
    return 0;
}
