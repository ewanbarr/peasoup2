#ifndef PEASOUP_FREQUENCYSERIES_CUH
#define PEASOUP_FREQUENCYSERIES_CUH

#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {

	struct FrequencySeriesMetaData: public MetaData
	{
	    float binwidth;
	    float dm;
	    float acc;
	    float nn;
	};
	
	template <System system, typename ValueType>
	using FrequencySeries = Container<system,ValueType,FrequencySeriesMetaData>;
    }
} 



#endif //PEASOUP_FREQUENCYSERIES_CUH
