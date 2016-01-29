#ifndef PEASOUP_FREQUENCYSERIES_CUH
#define PEASOUP_FREQUENCYSERIES_CUH

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
	    
	    void display()
	    {
		utils::print("----------------------------\n",
			     __PRETTY_FUNCTION__,"\n",
			     "Bin width: ",binwidth," Hz\n",
			     "DM: ",dm," pccm^-3\n",
			     "Acceleration: ",acc," m/s/s\n",
			     "Nearest neighbour: ",nn,"\n",
			     "----------------------------\n");
	    }
	    
	};
	
	template <System system, typename ValueType>
	using FrequencySeries = Container<system,ValueType,FrequencySeriesMetaData>;
    }
} 



#endif //PEASOUP_FREQUENCYSERIES_CUH
