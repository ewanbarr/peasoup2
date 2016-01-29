#ifndef PEASOUP_TIMESERIES_CUH
#define PEASOUP_TIMESERIES_CUH

#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {
	
	struct TimeSeriesMetaData: public MetaData
	{
	    float tsamp;
	    float dm;
	    float acc;
	    
	    void display()
	    {
		utils::print("----------------------------\n",
			     __PRETTY_FUNCTION__,"\n",
			     "Sampling time: ",tsamp," s\n",
			     "DM: ",dm," pccm^-3\n",
			     "Acceleration: ",acc," m/s/s\n",
			     "----------------------------\n");
	    }

	};
	
	template <System system, typename ValueType>
	using TimeSeries = Container<system,ValueType,TimeSeriesMetaData>;

    } //type
} //peasoup

#endif //PEASOUP_TIMESERIES_CUH
