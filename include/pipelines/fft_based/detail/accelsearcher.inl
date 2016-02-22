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
	    resampler->set_stream(stream);
	    r2cfft->set_stream(stream);
	    spectrum_former->set_stream(stream);
	    harmsum->set_stream(stream);
	    peak_finder->set_stream(stream);
	}

	template <>
	inline void AccelSearch<HOST>::set_stream(cudaStream_t stream)
	{
	    std::cerr << "Setting the stream has no effect on a HOST pipeline" << std::endl;
	}

	template <System system>
        void AccelSearch<system>::prepare()
	{
	    if (!args.acc_plan){
		if (args.user_acc_list.size() > 0){
		    args.acc_plan = std::make_shared<StaticAccelerationPlan>(args.user_acc_list);
		} else {
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
		utils::print("Processing acceleration: ",accel,"\n");
		resampler->set_accel(accel);
		resampler->execute();
		r2cfft->execute();
		spectrum_former->execute();
		harmsum->execute();
		peak_finder->execute();
		harmonic_still->distill(harm_dets,acc_dets);
		for (auto det: harm_dets)
		    {
			printf("F: %f  P: %f  S: %f  H: %d\n",det.freq,det.power,det.sigma,det.nh);
		    }
		harm_dets.clear();
	    }
	    acceleration_still->distill(acc_dets,dets);
	    printf("Final N cands: %d\n",dets.size());
	    for (auto det: acc_dets)
		{
		    printf("F: %f  P: %f  S: %f  H: %d\n",det.freq,det.power,det.sigma,det.nh);
		}
	    POP_NVTX_RANGE;
	}

    } // pipeline
} // peasoup
