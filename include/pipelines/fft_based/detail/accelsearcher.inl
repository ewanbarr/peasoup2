#include <algorithm>
#include <iostream>
#include "pipelines/fft_based/accelsearcher.cuh"
#include "utils/nvtx.hpp"


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
	    peak_finder = new PeakFinder<system,float>(spectrum,harmonics,dets,args.minsigma);
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
	    for (int ii=0;ii<args.acc_list.size();ii++)
		utils::print("Acc ",ii,": ",args.acc_list[ii],"\n");
	    
	    resampler->prepare();
	    r2cfft->prepare();
	    spectrum_former->prepare();
	    harmsum->prepare();
	    peak_finder->prepare();
	}

	template <System system>
        void AccelSearch<system>::run()
	{
	    PUSH_NVTX_RANGE(__PRETTY_FUNCTION__,1)
	    for (float accel: args.acc_list){
		utils::print("Processing acceleration: ",accel,"\n");
		resampler->set_accel(accel);
		resampler->execute();
		r2cfft->execute();
		spectrum_former->execute();
		harmsum->execute();
		peak_finder->execute();
	    }
	    POP_NVTX_RANGE
	}

    } // pipeline
} // peasoup
