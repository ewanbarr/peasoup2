#ifndef PEASOUP_DISPERSIONTIME_CUH
#define PEASOUP_DISPERSIONTIME_CUH

#include <vector>
#include "misc/system.cuh"
#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {
	
	struct DispersionTimeMetaData: public MetaData
	{
	    float tsamp;
	    std::vector<float> dms;
	};
	
	template <System system, typename ValueType>
	using DispersionTime = Container<system,ValueType,DispersionTimeMetaData>;

    } //type
} //peasoup

#endif //PEASOUP_DISPERSIONTIME_CUH
