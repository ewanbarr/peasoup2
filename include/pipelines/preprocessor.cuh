#ifndef PEASOUP_PREPROCESSOR_CUH
#define PEASOUP_PREPROCESSOR_CUH

#include <vector>
#include <utility>

#include "thrust/complex.h"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/fft.cuh"
#include "transforms/normaliser.cuh"
#include "transforms/baselinefinder.cuh"
#include "transforms/spectrumformer.cuh"
#include "transforms/zapper.cuh"

namespace peasoup {

    struct PeasoupArgs
    {
	std::vector<float> acc_list;
	std::vector<std::pair<float,float> > birdies;
    };

    namespace pipeline {

	using namespace type;
	using namespace transform;
	
	template <System system>
	class Preprocessor
	{
	public:
	    typedef thrust::complex<float> complex;
	    TimeSeries<system,float>& input;
	    TimeSeries<system,float>& output;
	    FrequencySeries<system,complex> fourier;
	    FrequencySeries<system,float> spectrum;
	    FrequencySeries<system,float> baseline;
	    Zapper<system,float>* zapper;
	    Normaliser<system,float>* normaliser;
	    SpectrumFormer<system,float>* spectrum_former;
	    BaselineFinder<system,float>* baseline_finder;
	    RealToComplexFFT<system>* r2cfft;
	    ComplexToRealFFT<system>* c2rfft;
	    PeasoupArgs& args;

	public:
	    Preprocessor(TimeSeries<system,float>& input,
			 TimeSeries<system,float>& output,
			 PeasoupArgs& args);
	    ~Preprocessor();
	    void prepare();
	    void run();
	};
    } //pipeline
} //peasoup

#include "pipelines/detail/preprocessor.inl"

#endif // PEASOUP_PREPROCESSOR_CUH
