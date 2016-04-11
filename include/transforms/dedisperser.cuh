#ifndef PEASOUP_DEDISPERSER_CUH
#define PEASOUP_DEDISPERSER_CUH

#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include "thirdparty/dedisp.h"

#include <data_types/timefrequency.cuh>
#include <data_types/dispersiontime.cuh>
#include "utils/logging.hpp"

namespace peasoup {
    namespace utils {

	inline void check_dedisp_error(dedisp_error error, std::string function_name);

    } // namespace utils
    
    namespace transform {
	
	/* 
	   Currently dedispersion is only implemented on the GPU.
	   Although a CPU version may one day be added (low priority),
	   we will not currently build out Host/Device derived base
	   classes to handle the different executions. 
	   Ideally the GPU dedispersion would take device vectors
	   but it is not currently set up to do this.
	*/
	
	class Dedisperser
	{
	private:
	    dedisp_plan plan;
	    type::TimeFrequencyBits<HOST>& input;
	    type::DispersionTime<HOST,uint8_t>& output;
	    std::vector<float> dm_list;
	    std::vector<int> chan_mask;
	    unsigned num_gpus;
	    	    
	public:
	    Dedisperser(type::TimeFrequencyBits<HOST>& input,
			type::DispersionTime<HOST,uint8_t>& output,
			unsigned num_gpus);
	    void set_dmlist(const std::vector<float>& dms);
	    const std::vector<float>& get_dmlist();
	    void set_chanmask(const std::vector<int>& mask);
	    void gen_dmlist(float dm_start, float dm_end,
			    float width, float tolerance);
	    void prepare();	    
	    void execute();
	};

    } // namespace transform
} // namespace peasoup

#include "transforms/detail/dedisperser.inl"

#endif // PEASOUP_DEDISPERSER_CUH
