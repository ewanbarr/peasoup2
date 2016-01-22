#ifndef PEASOUP_STREAM_CUH
#define PEASOUP_STREAM_CUH

#include <ios>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "utils/utils.cuh"

namespace peasoup {
    namespace io {

	class IOStream
	{
	private:
	    std::string handle;
	    
	public:
	    IOStream(std::string handle)
		:handle(handle){}
	    virtual void prepare()=0;
	    virtual void seekg(size_t offset, std::ios_base::seekdir way)=0;
	    virtual size_t tellg()=0;
	    virtual void read(char* buffer, size_t nbytes) = 0;
	    virtual void write(const char* buffer, size_t nbytes) = 0;
	    
	    std::string get_handle(){return handle;}
	    
	    template <class VectorType>
            void read_vector(VectorType& vec)
            {
                char* buffer = (char*) &(vec.data()[0]);
                size_t nbytes = vec.size() * sizeof(typename VectorType::value_type);
                this->read(buffer,nbytes);
            }
	    
	    template <class VectorType>
	    void write_vector(const VectorType& vec)
            {
                const char* buffer = (char*) &(vec.data()[0]);
                size_t nbytes = vec.size() * sizeof(typename VectorType::value_type);
                this->write(buffer,nbytes);
            }
	    
	};
	
    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
