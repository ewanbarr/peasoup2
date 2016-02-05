#include <sstream>
#include <iostream>
#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/fft.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"
#include "thrust/complex.h"

using namespace peasoup;

template<System system>
void fft_benchmark(size_t size, int nits=100)
{
    type::TimeSeries<system,float> input;
    type::FrequencySeries<system,thrust::complex<float> > output;
    transform::RealToComplexFFT<system> fft(input,output);
    utils::Timer clock;
    input.data.resize(size);
    input.metadata.tsamp = 0.000064;
    fft.prepare();
    clock.start();
    for (int ii=0;ii<nits;ii++)
	fft.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("RealToComplexFFT: ",size," pt: ",clock.elapsed()/nits," ms avg\n");
    std::cout << size << "\t" << clock.elapsed()/nits << "\n";
};

int main( int argc, char *argv[] )
{
    for (int ii=14;ii<27;ii++)
	fft_benchmark<DEVICE>(1<<ii);
    return 0;
}
