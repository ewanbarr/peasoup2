#include "pipelines/pipeline.cuh"

namespace peasoup {
    namespace pipeline {

	typedef typename type::TimeFrequencyBits<HOST> fil_type;
	typedef typename type::TimeSerues<HOST,float> tim_type;
	
	inline TimeFrequencyPipeline<::TimeFrequencyPipeline(fil_type& input, Options& args)
	    :input(input),args(args)
	{
	    dedisperser = new transform::Dedisperser(input,output,args.ngpus);
	    
	    
	    
	    dedisp.gen_dmlist(0.0,10.0,40.0,1.05);
	    const std::vector<float>& dm_list = dedisp.get_dmlist();
	    for (int ii=0;ii<dm_list.size()-1;ii++){
		ASSERT_TRUE(dm_list[ii]<10.0);
	    }
	    dedisp.prepare();
	    dedisp.execute();
	}
	
	

	
	



    } //pipeline
} // peasoup
