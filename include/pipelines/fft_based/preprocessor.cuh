#ifndef PEASOUP_PREPROCESSOR_CUH
#define PEASOUP_PREPROCESSOR_CUH

#include <vector>
#include <utility>
#include "cuda.h"

#include "thrust/complex.h"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/fft.cuh"
#include "transforms/normaliser.cuh"
#include "transforms/baselinefinder.cuh"
#include "transforms/spectrumformer.cuh"
#include "transforms/zapper.cuh"
#include "pipelines/args.hpp"
#include "transforms/pad.cuh"
#include "utils/logging.hpp"

namespace peasoup {

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
	    Pad<system,float>* padder;
	    Normaliser<system,float>* normaliser;
	    SpectrumFormer<system,float>* spectrum_former;
	    FDBaselineFinder<system,float>* baseline_finder;
	    RealToComplexFFT<system>* r2cfft;
	    ComplexToRealFFT<system>* c2rfft;
	    AccelSearchArgs& args;

	public:
	    Preprocessor(TimeSeries<system,float>& input,
			 TimeSeries<system,float>& output,
			 AccelSearchArgs& args);
	    ~Preprocessor();
	    void set_stream(cudaStream_t stream);
	    void prepare();
	    void run();
	};
    } //pipeline
} //peasoup

#include "pipelines/fft_based/detail/preprocessor.inl"

#endif // PEASOUP_PREPROCESSOR_CUH
