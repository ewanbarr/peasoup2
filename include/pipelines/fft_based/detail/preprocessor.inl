#include <algorithm>
#include <iostream>
#include "pipelines/fft_based/preprocessor.cuh"
#include "utils/nvtx.hpp"

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
        using namespace transform;
	
	template <System system>
	Preprocessor<system>::Preprocessor(TimeSeries<system,float>& input,
					   TimeSeries<system,float>& output,
					   AccelSearchArgs& args)
	    :input(input),output(output),args(args)
	{
	    float max_accel = *std::max_element(args.acc_list.begin(),args.acc_list.end());
	    if (max_accel<100) max_accel = 5000.0;
	    padder = new Pad<system,float>(input,input.data.size(),args.nfft);
	    r2cfft = new RealToComplexFFT<system>(input,fourier);
	    spectrum_former = new SpectrumFormer<system,float>(fourier,spectrum,false);
	    baseline_finder = new BaselineFinder<system,float>(spectrum,baseline,max_accel);
	    normaliser = new Normaliser<system,float>(fourier,fourier,baseline);
	    zapper = new Zapper<system,float>(fourier,args.birdies);
	    c2rfft = new ComplexToRealFFT<system>(fourier,output);
	}
	
	template <System system>
        Preprocessor<system>::~Preprocessor()
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
	inline void Preprocessor<DEVICE>::set_stream(cudaStream_t stream)
	{
	    padder->set_stream(stream);
	    r2cfft->set_stream(stream);
            spectrum_former->set_stream(stream);
            baseline_finder->set_stream(stream);
            normaliser->set_stream(stream);
            zapper->set_stream(stream);
            c2rfft->set_stream(stream);
	}

	template <>
	inline void Preprocessor<HOST>::set_stream(cudaStream_t stream)
	{
	    std::cerr << "Setting the stream has no effect on a HOST pipeline" << std::endl;
	}

	template <System system>
        void Preprocessor<system>::prepare()
	{
	    padder->prepare();
	    r2cfft->prepare();
	    spectrum_former->prepare();
	    baseline_finder->prepare();
	    normaliser->prepare();
	    zapper->prepare();
	    c2rfft->prepare();
	}

	template <System system>
        void Preprocessor<system>::run()
	{
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
