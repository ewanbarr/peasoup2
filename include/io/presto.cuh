#ifndef PEASOUP_STREAM_CUH
#define PEASOUP_STREAM_CUH

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "utils/utils.cuh"

namespace peasoup {
    namespace io {

	class PrestoParser
	{
	private:
            InputStream* data_stream;
	    InputStream* metadata_stream;
	    
        public:
	    PrestoParser(){}
            template <typename DataType> void read(DataType& data);
        };
	
	

    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
