#ifndef PEASOUP_ACCELSEARCHER_CUH
#define PEASOUP_ACCELSEARCHER_CUH

#include <vector>
#include "cuda.h"
#include "data_types/candidates.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "data_types/harmonicseries.cuh"
#include "transforms/resampler.cuh"
#include "transforms/fft.cuh"
#include "transforms/spectrumformer.cuh"
#include "transforms/harmonicsum.cuh"
#include "transforms/peakfinder.cuh"
#include "pipelines/args.hpp"
#include "utils/printer.hpp"

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
			AccelSearchArgs& args);
	    ~AccelSearch();
	    void set_stream(cudaStream_t stream);
	    void prepare();
	    void run();
	    
	};
 
    } //pipeline
} //peasoup

#include "pipelines/fft_based/detail/accelsearcher.inl"

#endif // PEASOUP_ACCELSEARCHER_CUH