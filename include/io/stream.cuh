#ifndef PEASOUP_STREAM_CUH
#define PEASOUP_STREAM_CUH

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "utils/utils.cuh"

namespace peasoup {
    namespace io {
	namespace internal { 

	class IOStream
	{
	private:
	    std::string handle;
	    
	public:
	    IOStream(std::string handle)
		:handle(handle){}
	    virtual void prepare()=0;
	    virtual void seek(size_t offset, ios_base::seekdir way)=0;
	    virtual void tell()=0;
	    virtual void read(char* buffer, size_t nbytes) = 0;
	    virtual void write(const char* buffer, size_t nbytes) = 0;

	    template <class VectorType>
            void read_vector(VectorType& vec)
            {
                char* buffer = (char*) &(vec.data()[0]);
                size_t nbytes = vec.size() * sizeof(VectorType::value_type);
                this->read(buffer,nbytes);
            }
	    
	    void write_vector(const VectorType& vec)
            {
                const char* buffer = (char*) &(vec.data()[0]);
                size_t nbytes = vec.size() * sizeof(VectorType::value_type);
                this->write(buffer,nbytes);
            }
	    
	};
	
    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
