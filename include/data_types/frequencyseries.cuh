#ifndef PEASOUP_FREQUENCYSERIES_CUH
#define PEASOUP_FREQUENCYSERIES_CUH

#include <string>
#include <sstream>
#include "data_types/metadata.cuh"
#include "data_types/container.cuh"
#include "utils/printer.hpp"

namespace peasoup {
    namespace type {

	struct FrequencySeriesMetaData: public MetaData
	{
	    float binwidth;
	    float dm;
	    float acc;
	    bool nn;
	    
	    std::string display()
	    {
		std::stringstream stream;
		stream << "Bin width: "<<binwidth<<" Hz\n"<<
		    "DM: "<<dm<<" pccm^-3\n"<<
		    "Acceleration: "<<acc<<" m/s/s\n"<<
		    "Nearest neighbour: "<<nn<<"\n";
		return stream.str();
	    }
	    
	    FrequencySeriesMetaData()
		:binwidth(0.0),
		 dm(0.0),
		 acc(0.0),
		 nn(false){}
	    
	};
	
	template <System system, typename ValueType>
	using FrequencySeries = Container<system,ValueType,FrequencySeriesMetaData>;
    }
} 



#endif //PEASOUP_FREQUENCYSERIES_CUH
