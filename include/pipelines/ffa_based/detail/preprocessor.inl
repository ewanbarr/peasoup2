#include <algorithm>
#include <iostream>
#include "pipelines/ffa_based/preprocessor.cuh"
#include "utils/nvtx.hpp"

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
        using namespace transform;
	
	template <System system>
	FFAPreprocessor<system>::FFAPreprocessor(TimeSeries<system,float>& input,
						 TimeSeries<system,float>& output,
						 FFASearchArgs& args)
	    :input(input),output(output),args(args)
	{
	    baseline_finder = new TDBaselineFinder<system,float>(input,baseline,args.max_smoothing);
	    //normaliser = new Normaliser<system,float>(fourier,fourier,baseline);
	}
	
	template <System system>
        FFAPreprocessor<system>::~FFAPreprocessor()
	{
	    delete r2cfft;
	    delete spectrum_former;
	    delete baseline_finder;
	    delete normaliser;
	    delete zapper;
	    delete c2rfft;
	    delete padder;
	}

	template <>
	inline void FFAPreprocessor<DEVICE>::set_stream(cudaStream_t stream)
	{
	    LOG(logging::get_logger("pipeline.preprocessor"),logging::DEBUG,
		"Setting stream on transforms (stream: ",stream,")");
	    padder->set_stream(stream);
	    r2cfft->set_stream(stream);
            spectrum_former->set_stream(stream);
            baseline_finder->set_stream(stream);
            normaliser->set_stream(stream);
            zapper->set_stream(stream);
            c2rfft->set_stream(stream);
	}

	template <>
	inline void FFAPreprocessor<HOST>::set_stream(cudaStream_t stream)
	{
	    LOG(logging::get_logger("pipeline.preprocessor"),logging::WARNING,
		"Setting the stream has no effect on a HOST pipeline");
	}

	template <System system>
        void FFAPreprocessor<system>::prepare()
	{
	    LOG(logging::get_logger("pipeline.preprocessor"),logging::DEBUG,
		"Preparing preprocessor pipeline\n",
		"Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    padder->prepare();
	    r2cfft->prepare();
	    spectrum_former->prepare();
	    baseline_finder->prepare();
	    normaliser->prepare();
	    zapper->prepare();
	    c2rfft->prepare();
	    LOG(logging::get_logger("pipeline.preprocessor"),logging::DEBUG,
                "Prepared preprocessor pipeline\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
	}

	template <System system>
        void FFAPreprocessor<system>::run()
	{
	    LOG(logging::get_logger("pipeline.preprocessor"),logging::DEBUG,
		"Executing preprocessor pipeline");
	    PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1)
            padder->execute();	
	    r2cfft->execute();
	    spectrum_former->execute();
	    baseline_finder->execute();
	    normaliser->execute();
	    zapper->execute();
	    c2rfft->execute();
	    POP_NVTX_RANGE
	}

    } // pipeline
} // peasoup
