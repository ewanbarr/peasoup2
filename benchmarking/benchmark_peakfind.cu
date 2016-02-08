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
#include "thrust/scan.h"

using namespace peasoup;

template<System system>
void peakfind_benchmark(size_t size, size_t npeaks, int nharms, int nits=100)
{
    int step = size/npeaks;
    type::FrequencySeries<HOST,float> hinput;
    hinput.data.resize(size);
    for (int ii=0;ii<size;ii+=step)
	hinput.data[ii] = 110000.0;

    type::HarmonicSeries<HOST,float> hharms;
    hharms.data.resize(size*nharms);
    for (int ii=0;ii<size*nharms;ii+=step)
        hharms.data[ii] = 110000.0;

    type::FrequencySeries<system,float> input=hinput;
    input.metadata.binwidth = 1/600.0;
    input.metadata.dm = 235.3;
    input.metadata.acc = 0.33;
   
    type::HarmonicSeries<system,float> harms = hharms;
    transform::HarmonicSum<system,float> sum(input,harms,nharms);
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
    std::cout << size << "\t" << npeaks << "\t" << nharms << "\t" <<clock.elapsed()/nits << "\n";
};

int main( int argc, char *argv[] )
{
    if (argc == 5)
	{
	    size_t size = (size_t) atoi(argv[1]);
	    size_t npeaks = (size_t) atoi(argv[2]);
	    int nharms = (int) atoi(argv[3]);
	    System system = static_cast<System>(atoi(argv[4]));
	    if (system == HOST)
		peakfind_benchmark<HOST>(size,npeaks,nharms,1);
	    else
		peakfind_benchmark<DEVICE>(size,npeaks,nharms,1);
	} 
    else
	{
	    for (int ii=20;ii<27;ii++){
		for (int nharms=1;nharms<6;nharms++)
		    peakfind_benchmark<DEVICE>(1<<ii,(size_t)((1<<ii)/1000.0),nharms,1);
	    }
	}
    return 0;
}
