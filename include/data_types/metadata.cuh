#ifndef PEASOUP_METADATA_CUH
#define PEASOUP_METADATA_CUH

#include <string>
#include "utils/printer.hpp"

namespace peasoup {
    
    struct MetaData 
    {
	virtual std::string display()=0;
    };

} 

#endif //PEASOUP_METADATA_H
