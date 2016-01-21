#ifndef PEASOUP_TIMEFREQUENCY_CUH
#define PEASOUP_TIMEFREQUENCY_CUH

#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {
	
	struct TimeFrequencyMetaData: public MetaData
	{
	    float tsamp;
	    unsigned nchans;
	    float foff;
	    float fch1;
	};
	
	
	template <System system, typename ValueType>
	class TimeFrequency: public Container<system,ValueType,TimeFrequencyMetaData>
	{
	private:
	    typedef Container<system,ValueType,TimeFrequencyMetaData> Parent;
	public:
	    using Parent::Container;
	    using Parent::data;
	    using Parent::metadata;
	    size_t get_nsamps(){return data.size()/(metadata.nchans);}
	};

	template <System system>
	class TimeFrequencyBits: public Container<system,uint8_t,TimeFrequencyMetaData>
	{
	private:
	    typedef Container<system,uint8_t,TimeFrequencyMetaData> Parent;
	    unsigned bits_per_byte;

	public:
	    using Parent::data;
            using Parent::metadata;
	    unsigned nbits;

	    TimeFrequencyBits(unsigned nbits=0)
		:Container<system,uint8_t,TimeFrequencyMetaData>(),nbits(nbits),bits_per_byte(8/nbits){}
	    size_t get_nsamps(){return data.size()*bits_per_byte/(metadata.nchans);}
	};
	
	
    } //type
} //peasoup

#endif //PEASOUP_TIMEFREQUENCY_CUH
