#include <sstream>
#include <iostream>
#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/harmonicsum.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"
#include "utils/timer.cuh"

using namespace peasoup;

template<System system>
void hsum_benchmark(size_t size, unsigned nharms, int nits=100, bool use_default=false)
{
    type::FrequencySeries<system,float> input;
    type::HarmonicSeries<system,float> output;
    transform::HarmonicSum<system,float> summer(input,output,nharms,use_default);
    utils::Timer clock;
    input.data.resize(size);
    input.metadata.binwidth = 0.001;
    summer.prepare();
    clock.start();
    for (int ii=0;ii<nits;ii++)
	summer.execute();
    utils::check_cuda_error(__PRETTY_FUNCTION__);
    clock.stop();
    utils::print("HarmonicSum: ",size," pt, ",nharms," harmonics: ",clock.elapsed()/nits," ms avg\n");
    std::cout << size << "\t" << nharms << "\t" << clock.elapsed()/nits << "\n";

};

int main( int argc, char *argv[] )
{
    if (argc==4)
	{
	    size_t size =(size_t)atoi(argv[1]);
	    int nharm = atoi(argv[2]);
	    bool use_default = (bool) atoi(argv[3]);
	    hsum_benchmark<DEVICE>(size,nharm,100,use_default);
	}
    else
	{
	    std::cout << "#\n#-------------------fast math--------------------\n#\n";

	    for (int n=1;n<6;n++){
		for (int ii=10;ii<27;ii++)
		    hsum_benchmark<DEVICE>(1<<ii,n);
	    }
	    
	    std::cout << "#\n#-------------------only defualt--------------------\n#\n";
	    
	    for (int n=1;n<6;n++){
                for (int ii=10;ii<27;ii++)
                    hsum_benchmark<DEVICE>(1<<ii,n,100,true);
            }



	}
    return 0;
}
