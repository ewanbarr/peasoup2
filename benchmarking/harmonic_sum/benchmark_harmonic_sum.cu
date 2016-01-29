#include "utils/nvtx.hpp"
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/harmonicsum.cuh"
#include "utils/utils.cuh"
#include "utils/printer.hpp"

using namespace peasoup;

template <System system>
class HarmSumBenchmark
{
public:
    type::FrequencySeries<system,float> input;
    type::HarmonicSeries<system,float> output;
    transform::HarmonicSum<system,float> summer;
    size_t size;
    void _setup(){
	input.data.resize(size);
	input.metadata.binwidth = 0.001;
	summer.prepare();
    }
    void _teardown(){};
    

public:
    HarmSumBenchmark(size_t size,int nharms)
	:size(size),summer(input,output,nharms){}
    
    void run()
    {
	_setup();
	utils::print("Running harmonic sum test for 100 iterations\n");
	PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1);
	for (int ii=0;ii<10;ii++)
	    summer.execute();
	utils::check_cuda_error(__PRETTY_FUNCTION__);
	POP_NVTX_RANGE;    
	_teardown();
    }
};


int main()
{
    HarmSumBenchmark<HOST> host(1<<22,4);
    host.run();
    
    HarmSumBenchmark<DEVICE> device(1<<22,4);
    device.run();
    
    return 0;
}
