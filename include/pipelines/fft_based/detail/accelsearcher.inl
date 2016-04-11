#include <algorithm>
#include <iostream>
#include "pipelines/fft_based/accelsearcher.cuh"
#include "utils/nvtx.hpp"
#include "utils/utils.cuh"

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
        using namespace transform;
	
	template <System system>
	AccelSearch<system>::AccelSearch(TimeSeries<system,float>& input,
					 std::vector<Detection>& dets,
					 AccelSearchArgs& args)
	    :input(input), dets(dets), args(args)
	{
	    resampler = new TimeDomainResampler<system,float>(input,timeseries_r);
	    r2cfft = new RealToComplexFFT<system>(timeseries_r,fourier);
	    spectrum_former = new SpectrumFormer<system,float>(fourier,spectrum,true);
	    harmsum = new HarmonicSum<system,float>(spectrum,harmonics,args.nharm);
	    peak_finder = new PeakFinder<system,float>(spectrum,harmonics,harm_dets,args.minsigma);
	    harmonic_still = new HarmonicDistiller(args.hdistill_tol,args.nharm);
	    float tobs = input.data.size() * input.metadata.tsamp;
	    acceleration_still = new AccelerationDistiller(tobs,args.adistill_tol);
	}
	
	template <System system>
        AccelSearch<system>::~AccelSearch()
	{
	    delete resampler;
	    delete r2cfft;
	    delete spectrum_former;
	    delete harmsum;
	    delete peak_finder;
	}

	template <>
	inline void AccelSearch<DEVICE>::set_stream(cudaStream_t stream)
	{
	    LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
		"Setting stream on transforms (stream: ",stream,")");
	    resampler->set_stream(stream);
	    r2cfft->set_stream(stream);
	    spectrum_former->set_stream(stream);
	    harmsum->set_stream(stream);
	    peak_finder->set_stream(stream);
	}

	template <>
	inline void AccelSearch<HOST>::set_stream(cudaStream_t stream)
	{
	    LOG(logging::get_logger("pipeline.accelsearch"),logging::WARNING,
		"Setting the stream has no effect on a HOST pipeline");	    
	}
	
	template <System system>
        void AccelSearch<system>::prepare()
	{
	    LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
                "Preparing AccelSearch\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    
	    if (!args.acc_plan){
		LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
		    "No acc_plan set in accelsearch args");
		if (args.user_acc_list.size() > 0){
		    LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
			"Setting up StaticAccelerationPlan with ",args.user_acc_list.size(),
			" user defined accelerations");
		    args.acc_plan = std::make_shared<StaticAccelerationPlan>(args.user_acc_list);
		} else {
		    LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
                        "Setting up DMDependentAccelerationPlan");
		    args.acc_plan = std::make_shared<DMDependentAccelerationPlan>
			(args.acc_start,args.acc_end,args.acc_tol,args.acc_pulse_width,
			 input.data.size(),input.metadata.tsamp, args.cfreq, args.chbw);
		}
	    }
	    resampler->prepare();
	    r2cfft->prepare();
	    spectrum_former->prepare();
	    harmsum->prepare();
	    peak_finder->prepare();
	}

	template <System system>
        void AccelSearch<system>::run()
	{
	    PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1);
	    std::vector<type::Detection> acc_dets;
	    args.acc_plan->get_accelerations(acc_list,input.metadata.dm);
	    for (float accel: acc_list){
		LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
		    "Processing acceleration: ",accel," m/s/s");
		resampler->set_accel(accel);
		resampler->execute();
		r2cfft->execute();
		spectrum_former->execute();
		harmsum->execute();
		peak_finder->execute();
		harmonic_still->distill(harm_dets,acc_dets);
		LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
		    "Pre harmonic distilled candidates\n",type::detections_as_string(harm_dets));
		harm_dets.clear();
	    }
	    acceleration_still->distill(acc_dets,dets);
	    LOG(logging::get_logger("pipeline.accelsearch"),logging::DEBUG,
		"Pre acceleration distilled candidates\n",type::detections_as_string(acc_dets));
	    
	    POP_NVTX_RANGE;
	}

    } // pipeline
} // peasoup
