#idndef PEASOUP_ACCELSEARCHER_CUH
#define PEASOUP_ACCELSEARCHER_CUH

#include <algorithm>
#include <>

namespace peasoup {
    namespace pipeline {
	
	using namespace type;
	using namespace transform;

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
	    TimeSeries<system,float>& input;
	    TimeSeries<system,float> timeseries_r;
	    FrequencySeries<system,complex> fourier;
	    FrequencySeries<system,float> spectrum;
	    HarmonicSeries<system,float> harmonics;
	    //Peak list
	    TimeDomainResampler<system,float>* resampler;
	    RealToComplexFFT<system>* r2cfft;
	    SpectrumFormer<system,float>* spectrum_former;
	    HarmonicSum<system,float>* harmsum;
	    PeakFinder<system,float>* peak_finder; //<--needs to be written
	    AccelSearchArgs& args;
	    
	public:
	    
	    AccelSearch(TimeSeries<system,float>& input,
			AccelSearchArgs& args)
		:input(input), args(args)
	    {
		resampler = new TimeDomainResampler<system,float>(input,timeseries_r);
		r2cfft = new RealToComplexFFT<system>(timeseries_r,fourier);
		spectrum_former = new SpectrumFormer<system,float>(fourier,spectrum);
		harmsum = new HarmonicSum<system,float>(spectrum,harmonics);
		peak_finder = new PeakFinder<system,float>(spectrum,harmonics,peaks);
	    }
	    
	    
	};

 
	} //worker
    } //pipeline
} //peasoup


#endif // PEASOUP_ACCELSEARCHER_CUH
