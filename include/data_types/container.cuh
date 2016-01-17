#ifndef PEASOUP_CONTAINER_CUH
#define PEASOUP_CONTAINER_CUH

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "misc/system.cuh"

namespace peasoup {


    
    template <System system, typename ValueType, typename MetaDataType >
    class Container
    {
    public:
	typedef typename SystemVector<system,ValueType>::vector_type vector_type;
	typedef ValueType value_type;
	typedef MetaDataType metadata_type;
	vector_type data;
	metadata_type metadata;

	Container(){}

        Container(const Container &v)
            :data(v.data),metadata(v.metadata) {}

        template <System other_system>
        Container(const Container<other_system,ValueType,MetaDataType> &v)
            :data(v.data),metadata(v.metadata) {}

        template <System other_system>
        Container &operator=(const Container<other_system,ValueType,MetaDataType> &v)
        { data=v.data; metadata=v.metadata; return *this; }

    };

} // namespace peasoup

#endif

