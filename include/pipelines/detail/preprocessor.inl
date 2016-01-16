#include "pipelines/preprocessor.cuh"
#include <algorithm>

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
        using namespace transform;
	
	template <System system>
	Preprocessor<system>::Preprocessor(TimeSeries<system,float>& input,
					   TimeSeries<system,float>& output,
					   PeasoupArgs args)
	    :input(input),output(output),args(args)
	{
	    float max_accel = *std::max_element(args.acc_list.begin(),args.acc_list.end());
	    if (max_accel<100) max_accel = 100.0;
	    r2cfft = new RealToComplexFFT<system>(input,fourier);
	    spectrum_former = new SpectrumFormer<system,float>(fourier,spectrum);
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
	}

	template <System system>
        void Preprocessor<system>::prepare()
	{
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
	    r2cfft->execute();
	    spectrum_former->form();
	    baseline_finder->find_baseline();
	    normaliser->normalise();
	    zapper->execute();
	    c2rfft->execute();
	}

    } // pipeline
} // peasoup
