#include "transforms/dedisperser.cuh"

#define DEDISP_SAFE_CALL(call) \
    (utils::check_dedisp_error(call,__PRETTY_FUNCTION__))

namespace peasoup {

    inline void utils::check_dedisp_error(dedisp_error error, 
					  std::string function_name)
    {
	if (error != DEDISP_NO_ERROR){
	    std::stringstream error_msg;
	    error_msg << function_name
		      << " failed with DEDISP error: "
		      << dedisp_get_error_string(error)
		      << std::endl;
	    LOG(logging::get_logger("transform.dedisperser"),logging::ERROR,error_msg.str());
	    throw std::runtime_error(error_msg.str());
	}
    }

    namespace transform {
	
	Dedisperser::Dedisperser(type::TimeFrequencyBits<HOST>& input,
				 type::DispersionTime<HOST,uint8_t>& output,
				 unsigned num_gpus)
	    :input(input),output(output),num_gpus(num_gpus),
	     chan_mask(std::vector<int>(input.metadata.nchans,1))
	{
	    LOG(logging::get_logger("transform.dedisperser"),logging::DEBUG,
		"Creating Dedisp plan");
            DEDISP_SAFE_CALL(dedisp_create_plan_multi(
	          &plan, input.metadata.nchans, input.metadata.tsamp,
	          input.metadata.fch1, input.metadata.foff, num_gpus));
	}
	
	void Dedisperser::set_dmlist(const std::vector<float>& dms){dm_list = dms;}

	const std::vector<float>& Dedisperser::get_dmlist(){return dm_list;}
	
	void Dedisperser::set_chanmask(const std::vector<int>& mask)
	{
	    assert(mask.size()==input.metadata.nchans);
	    chan_mask = mask;
	}
	
	void Dedisperser::gen_dmlist(float dm_start, float dm_end,
				     float width, float tolerance)
	{
	    DEDISP_SAFE_CALL(dedisp_generate_dm_list(plan, dm_start, dm_end, width, tolerance));
	    const float* plan_dm_list = dedisp_get_dm_list(plan);
	    dm_list.assign(plan_dm_list,plan_dm_list + dedisp_get_dm_count(plan));
	}
	
	void Dedisperser::prepare()
	{
	    LOG(logging::get_logger("transform.dedisperser"),logging::DEBUG,
                "Preparing Dedisperser\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    
	    assert(dm_list.size() > 0);
	    DEDISP_SAFE_CALL(dedisp_set_dm_list(plan,&dm_list[0],dm_list.size()));
	    if (chan_mask.size()>0)
		DEDISP_SAFE_CALL(dedisp_set_killmask(plan,&chan_mask[0]));

	    size_t max_delay = dedisp_get_max_delay(plan);
	    size_t out_nsamps = input.get_nsamps()-max_delay;
	    output.data.resize(out_nsamps*dm_list.size());
	    output.metadata.dms = dm_list;
	    output.metadata.tsamp = input.metadata.tsamp;
	    LOG(logging::get_logger("transform.dedisperser"),logging::DEBUG,
                "Prepared Dedisperser\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples\n",
		"Number of DMs: ",dm_list.size(),"\n",
		"Using chan mask: ",chan_mask.size()!=0);
	}

	void Dedisperser::execute()
	{
	    LOG(logging::get_logger("transform.dedisperser"),logging::DEBUG,
                "Executing dedispersion");
	    DEDISP_SAFE_CALL(dedisp_execute(plan,input.get_nsamps(),
	        &(input.data[0]),input.nbits,&(output.data[0]),8,
		(unsigned)0));
	}


    } // namespace transform
} // namespace peasoup
