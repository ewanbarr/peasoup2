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
	    
	    std::string display()
            {
		std::stringstream stream;
		stream << "Bin widths: ";
		for (auto i:binwidths)
		    stream << i << ", ";
		stream << " Hz\n" << "DM: "<<dm<<" pccm^-3\n"
		       << "Acceleration: "<<acc<<" m/s/s\n"
		       << "Nearest neighbour: "<<nn<<"\n";
		return stream.str();	 
            }
	    
	    HarmonicSeriesMetaData()
		:dm(0.0),
		 acc(0.0),
		 nn(false){}
	};
	
	template <System system, typename ValueType>
	using HarmonicSeries = Container<system,ValueType,HarmonicSeriesMetaData>;
    
    } //namespace type
} //namespace peasoup

#endif //PEASOUP_HARMONICSERIES_CUH
