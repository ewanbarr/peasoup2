#ifndef PEASOUP_STREAM_CUH
#define PEASOUP_STREAM_CUH

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "io/stream.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace io {

	template <class StreamType>
	void safe_open(StreamType& infile, std::string filename, ios_base::openmode mode)
	{
	    infile.open(filename.c_str(),mode);
	    utils::check_file_error(infile,filename);
	}
	
	class FileStream: public IOStream
	{
	private:
	    std::ifstream file;
	    std::string filename;
	    ios_base::openmode mode;

	public:
	    FileStream(std::string filename, ios_base::openmode mode = ios_base::in | std::ifstream::binary)
		:IOStream(filename),filename(filename),mode(mode){}
	    ~FileStream(){ file.close(); }
	    void prepare(){ safe_open<std::ifstream>(file, filename, mode); }
	    void read(char* buffer, size_t nbytes){file.read(buffer,nbytes);}
	    void write(const char* buffer, size_t nbytes){ file.write(buffer,nbytes); }
	    void seek(size_t offset, ios_base::seekdir way){file.seekg(offset,way);}
            void tell(){return file.tellg();}
	};

    } // io
} // peasoup

#endif // PEASOUP_STREAM_CUH
