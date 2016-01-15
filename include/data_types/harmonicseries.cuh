#ifndef PEASOUP_HARMONICSERIES_CUH
#define PEASOUP_HARMONICSERIES_CUH

#include <vector>
#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {

	struct HarmonicSeriesMetaData: public MetaData
	{
	    std::vector<float> binwidths;
	    float dm;
	    float acc;
	};
	
	template <System system, typename ValueType>
	using HarmonicSeries = Container<system,ValueType,HarmonicSeriesMetaData>;
    
    } //namespace type
} //namespace peasoup

#endif //PEASOUP_HARMONICSERIES_CUH
