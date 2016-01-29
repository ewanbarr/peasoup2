#include "pipelines/pipeline_builder.cuh"
#include "io/file.cuh"
#include "io/sigproc.cuh"

namespace peasoup {
    namespace pipeline {
	
	PipelineBuilder::build(cmdline::CmdLineOptions& args)
	{
	    if (args.format == "sigproc"){
		io::IOStream* stream = new io::FileStream(args.infilename);
		stream->prepare();
		auto data_type = io::sigproc::get_data_type(stream);
		if (data_type == TIMESERIES)
		    {
			typedef type::TimeSeries<HOST,float> data_type;
			data_type data;
			io::sigproc::SigprocReader< data_type > reader(data,stream);
			reader.read();
			_launch_timeseries_pipeline(data,args);
		    }
		else 
		    {
			typedef type::TimeFrequencyBits<HOST> data_type;
			data_type data(0);
			io::sigproc::SigprocReader< data_type > reader(data,stream);
			reader.read();
			_launch_timefrequency_pipeline(data,args);
		    }
	    }
	}
	
	PipelineBuilder::_launch_timeseries_pipeline(type::TimeSeries<HOST,float>& data,
						     cmdline::CmdLineOptions& args)
	{

	}


	PipelineBuilder::_launch_imefrequency_pipeline(type::TimeFrequencyBits<HOST>& data,
						       cmdline::CmdLineOptions& args)
	{

        }


	
    } // pipeline
} // peasoup
