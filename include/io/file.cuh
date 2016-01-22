#ifndef PEASOUP_FILESTREAM_CUH
#define PEASOUP_FILESTREAM_CUH

#include <ios>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "io/stream.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace io {

	void safe_file_open(std::fstream& file, std::string filename, std::ios_base::openmode mode)
	{
	    file.open(filename.c_str(),mode);
	    if(!file.good()) {
		std::stringstream error_msg;
                error_msg << "File "<< filename << " could not be opened: ";
                if ( (file.rdstate() & std::fstream::failbit ) != 0 )
                    error_msg << "Logical error on i/o operation" << std::endl;
                if ( (file.rdstate() & std::fstream::badbit ) != 0 )
                    error_msg << "Read/writing error on i/o operation" << std::endl;
                if ( (file.rdstate() & std::fstream::eofbit ) != 0 )
                    error_msg << "End-of-File reached on input operation" << std::endl;
                throw std::runtime_error(error_msg.str());
            }
	}
	
	class FileStream: public IOStream
	{
	private:
	    std::fstream file;
	    std::string filename;
	    std::ios_base::openmode mode;
	    
	public:
	    FileStream(std::string filename, std::ios_base::openmode mode = std::ios_base::in | std::ifstream::binary)
		:IOStream(filename),filename(filename),mode(mode){}
	    ~FileStream(){ file.close(); }
	    void prepare(){ safe_file_open(file, filename, mode); }
	    void read(char* buffer, size_t nbytes){file.read(buffer,nbytes);}
	    void write(const char* buffer, size_t nbytes){ file.write(buffer,nbytes); }
	    void seekg(size_t offset, std::ios_base::seekdir way){file.seekg(offset,way);}
            size_t tellg(){return file.tellg();}
	};

    } // io
} // peasoup

#endif // PEASOUP_FILESTREAM_CUH
