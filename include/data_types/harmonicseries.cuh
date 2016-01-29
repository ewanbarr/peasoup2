#ifndef PEASOUP_HARMONICSERIES_CUH
#define PEASOUP_HARMONICSERIES_CUH

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include "data_types/metadata.cuh"
#include "data_types/container.cuh"

namespace peasoup {
    namespace type {

	struct HarmonicSeriesMetaData: public MetaData
	{
	    std::vector<float> binwidths;
	    float dm;
	    float acc;
	    bool nn;
	    
	    void display()
            {
		std::stringstream widths;
		for (auto i:binwidths)
		    widths << i << ", ";
		utils::print("----------------------------\n",
			     __PRETTY_FUNCTION__,"\n",
                             "Bin widths: ",widths.str()," Hz\n",
                             "DM: ",dm," pccm^-3\n",
                             "Acceleration: ",acc," m/s/s\n",
                             "Nearest neighbour: ",nn,"\n",
			     "----------------------------\n");
            }

	};
	
	template <System system, typename ValueType>
	using HarmonicSeries = Container<system,ValueType,HarmonicSeriesMetaData>;
    
    } //namespace type
} //namespace peasoup

#endif //PEASOUP_HARMONICSERIES_CUH
