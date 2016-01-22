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
	class DispersionTime: public Container<system,ValueType,DispersionTimeMetaData>
	{
	private:
            typedef Container<system,ValueType,DispersionTimeMetaData> Parent;

        public:
            using Parent::data;
            using Parent::metadata;
	    using Parent::Container;
	    size_t get_nsamps(){return data.size()/metadata.dms.size();}
	};

    } //type
} //peasoup

#endif //PEASOUP_DISPERSIONTIME_CUH
