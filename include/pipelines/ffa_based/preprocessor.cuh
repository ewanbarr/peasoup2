#ifndef PEASOUP_FFAPREPROCESSOR_CUH
#define PEASOUP_FFAPREPROCESSOR_CUH

#include <vector>
#include <utility>
#include "cuda.h"

#include "thrust/complex.h"
#include "data_types/timeseries.cuh"
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
	class FFAPreprocessor
	{
	public:
	    TimeSeries<system,float>& input;
	    TimeSeries<system,float>& output;
	    TimeSeries<system,float> baseline;
	    TDBaselineFinder<system,float>* baseline_finder;
	    Normaliser<system,float>* normaliser;
	    FFASearchArgs& args;

	public:
	    FFAPreprocessor(TimeSeries<system,float>& input,
			    TimeSeries<system,float>& output,
			    FFASearchArgs& args);
	    ~FFAPreprocessor();
	    void set_stream(cudaStream_t stream);
	    void prepare();
	    void run();
	};
    } //pipeline
} //peasoup

#include "pipelines/ffa_based/detail/preprocessor.inl"

#endif // PEASOUP_PREPROCESSOR_CUH
