#ifndef PEASOUP_TIMESERIES_CUH
#define PEASOUP_TIMESERIES_CUH

#include <string>
#include <sstream>
#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {
	
	struct TimeSeriesMetaData: public MetaData
	{
	    float tsamp;
	    float dm;
	    float acc;
	    
	    std::string display()
	    {
		std::stringstream stream;
		stream << "Sampling time: "<<tsamp<<" s\n"
		       << "DM: "<<dm<<" pccm^-3\n"
		       << "Acceleration: "<<acc<<" m/s/s\n";
		return stream.str();
	    }
	    
	    TimeSeriesMetaData():tsamp(0.0),acc(0.0),dm(0.0){}
	};
	
	template <System system, typename ValueType>
	using TimeSeries = Container<system,ValueType,TimeSeriesMetaData>;

    } //type
} //peasoup

#endif //PEASOUP_TIMESERIES_CUH
