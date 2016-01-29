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

	    void display()
	    {
		utils::print("----------------------------\n",
			     __PRETTY_FUNCTION__,"\n",
			     "Sampling time: ",tsamp," s\n",
			     "Channel bandwidth: ",foff," MHz\n",
			     "Top frequency: ",fch1," MHz\n",
			     "Nchans: ",nchans,"\n",
			     "----------------------------\n");
	    }
	    
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

	public:
	    using Parent::data;
            using Parent::metadata;
	    unsigned nbits;

	    TimeFrequencyBits(unsigned nbits=0)
		:Container<system,uint8_t,TimeFrequencyMetaData>(),nbits(nbits){}
	    size_t get_nsamps(){return data.size()*(8/nbits)/(metadata.nchans);}
	};
	
	
    } //type
} //peasoup

#endif //PEASOUP_TIMEFREQUENCY_CUH
