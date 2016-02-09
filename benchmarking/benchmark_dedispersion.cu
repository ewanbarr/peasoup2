#include <vector>
#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/timefrequency.cuh"
#include "data_types/dispersiontime.cuh"
#include "transforms/dedisperser.cuh"
#include "utils/nvtx.hpp"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"


using namespace peasoup;

void fill_input(type::TimeFrequencyBits<HOST>& input, size_t nsamps)
{
    input.metadata.tsamp = 0.000064;
    input.metadata.nchans = 1024;
    input.metadata.foff = -0.390;
    input.metadata.fch1 = 1510.0;
    uint8_t bits_per_byte = 8/input.nbits;
    input.data.assign(input.metadata.nchans*nsamps/bits_per_byte,0);
}

void benchmark(size_t nsamps, uint8_t nbits, int ndms, int ngpus)
{
    //Build input
    type::TimeFrequencyBits<HOST> input(nbits);
    fill_input(input,nsamps);
    type::DispersionTime<HOST,uint8_t> output;
    transform::Dedisperser dedisp(input,output,ngpus);
    std::vector<float> dm_list;
    utils::Timer clock;
    for (int ii=0;ii<ndms;ii++)
	dm_list.push_back(ndms/1000.0 * ii);
    dedisp.set_dmlist(dm_list);
    dedisp.prepare();
    clock.start();
    dedisp.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("Dedisperser: ",nsamps," pt, ",ndms," DMs: ",clock.elapsed()," ms avg\n");
    std::cout << nsamps << "\t" << ndms << "\t" << ngpus << "\t" << clock.elapsed() << "\n";
}

int main()
{
    size_t size = 1<<23;
    uint8_t nbits = 2;
    int ndms = 1000;
    benchmark(size,nbits,ndms,1);
    benchmark(size,nbits,ndms,2);
    benchmark(size,nbits,ndms,3);
    benchmark(size,nbits,ndms,4);
    benchmark(size,nbits,ndms,5);
    benchmark(size,nbits,ndms,6);
    benchmark(size,nbits,ndms,7);
    return 0;
}

