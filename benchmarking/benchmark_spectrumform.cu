#include <sstream>
#include <iostream>
#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/spectrumformer.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"
#include "thrust/complex.h"

using namespace peasoup;

template<System system>
void specform_benchmark(size_t size, bool nn, int nits=100)
{
    type::FrequencySeries<system,thrust::complex<float> > input;
    type::FrequencySeries<system,float> output;
    transform::SpectrumFormer<system,float> form(input,output,nn);
    utils::Timer clock;
    input.data.resize(size);
    form.prepare();
    clock.start();
    for (int ii=0;ii<nits;ii++)
	form.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("SpectrumFormer(",nn,"): ",size," pt: ",clock.elapsed()/nits," ms avg\n");
    std::cout << size << "\t" << nn << "\t" <<clock.elapsed()/nits << "\n";
};

int main( int argc, char *argv[] )
{
    if (argc == 3)
	{
	    size_t size = (size_t) atoi(argv[1]);
	    bool nn = (bool)atoi(argv[2]);
	    specform_benchmark<DEVICE>(size,nn);
	}
    else
	{
	    for (int ii=14;ii<27;ii++)
		specform_benchmark<DEVICE>(1<<ii,true);
	    for (int ii=14;ii<27;ii++)
                specform_benchmark<DEVICE>(1<<ii,false);
	}
    return 0;
}
