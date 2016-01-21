#ifndef PEASOUP_ACCELSEARCHER_CUH
#define PEASOUP_ACCELSEARCHER_CUH

#include <vector>
#include "data_types/candidates.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/resampler.cuh"
#include "transforms/fft.cuh"
#include "transforms/spectrumformer.cuh"
#include "transforms/harmonicsum.cuh"
#include "transforms/peakfinder.cuh"

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
	using namespace transform;

	struct AccelSearchArgs
	{
	    std::vector<float> acc_list;
	    float minsigma;
	    int nharm;
	};

	/* 
	   The AccelSearch class expects whitened data as input.
	   This can be obtained using a preprocessor pipeline.
	   The input time series should have a variance of ~1.
	   Frequency domain RFI excision should also be done in 
	   a preprocessor pipeline.
	*/
	template <System system>
	class AccelSearch
	{
	private:
	    typedef thrust::complex<float> complex;

	public:
	    TimeSeries<system,float>& input;
	    TimeSeries<system,float> timeseries_r;
	    FrequencySeries<system,complex> fourier;
	    FrequencySeries<system,float> spectrum;
	    HarmonicSeries<system,float> harmonics;
	    std::vector<Detection>& dets;
	    TimeDomainResampler<system,float>* resampler;
	    RealToComplexFFT<system>* r2cfft;
	    SpectrumFormer<system,float>* spectrum_former;
	    HarmonicSum<system,float>* harmsum;
	    PeakFinder<system,float>* peak_finder; 
	    AccelSearchArgs& args;
	    
	    AccelSearch(TimeSeries<system,float>& input,
			std::vector<Detection>& dets,
			AccelSearchArgs& args)
		:input(input), dets(dets), args(args)
	    {
		resampler = new TimeDomainResampler<system,float>(input,timeseries_r);
		r2cfft = new RealToComplexFFT<system>(timeseries_r,fourier);
		spectrum_former = new SpectrumFormer<system,float>(fourier,spectrum);
		harmsum = new HarmonicSum<system,float>(spectrum,harmonics,args.nharm);
		peak_finder = new PeakFinder<system,float>(spectrum,harmonics,dets,args.minsigma);
	    }

	    void prepare()
	    {
		resampler->prepare();
		r2cfft->prepare();
		spectrum_former->prepare();
		harmsum->prepare();
		peak_finder->prepare();
	    }
	    
	    void run()
	    {
		for (auto accel: args.acc_list){
		    resampler->resample(accel);
		    r2cfft->execute();
		    spectrum_former->form();
		    harmsum->sum();
		    peak_finder->execute();
		}
	    }
	    
	    
	};
 
    } //pipeline
} //peasoup


#endif // PEASOUP_ACCELSEARCHER_CUH
