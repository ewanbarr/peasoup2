#ifndef PEASOUP_UTILS_CUH
#define PEASOUP_UTILS_CUH

#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <execinfo.h>
#include <algorithm>
#include <iterator>

#include "cuda.h"
#include "cufft.h"
#include "cuda_runtime.h"

#include "misc/system.cuh"

namespace peasoup {
    namespace utils {

	inline
	void check_cuda_error(std::string msg="Unspecified location",cudaStream_t stream=NULL){
	    cudaError_t error;
	    if (stream == NULL)
		error = cudaDeviceSynchronize();
	    else
		error = cudaStreamSynchronize(stream);
	    if (error!=cudaSuccess){
		std::stringstream error_msg;
		error_msg << "CUDA failed with error: "
			  << cudaGetErrorString(error) << std::endl
			  << "Additional: " << msg << std::endl;
		throw std::runtime_error(error_msg.str());
	    }
	}
	
	inline
	void check_cufft_error(cufftResult error){
	    if (error!=CUFFT_SUCCESS){
		std::stringstream error_msg;
		error_msg << "cuFFT failed with error: ";
		switch (error)
		    {
		    case CUFFT_INVALID_PLAN:
			error_msg <<  "CUFFT_INVALID_PLAN";
			break;
			
		    case CUFFT_ALLOC_FAILED:
			error_msg <<  "CUFFT_ALLOC_FAILED";
			break;
			
		    case CUFFT_INVALID_TYPE:
			error_msg <<  "CUFFT_INVALID_TYPE";
			break;
			
		    case CUFFT_INVALID_VALUE:
			error_msg <<  "CUFFT_INVALID_VALUE";
			break;

		    case CUFFT_INTERNAL_ERROR:
			error_msg <<  "CUFFT_INTERNAL_ERROR";
			break;
			
		    case CUFFT_EXEC_FAILED:
			error_msg <<  "CUFFT_EXEC_FAILED";
			break;

		    case CUFFT_SETUP_FAILED:
			error_msg <<  "CUFFT_SETUP_FAILED";
			break;
			
		    case CUFFT_INVALID_SIZE:
			error_msg <<  "CUFFT_INVALID_SIZE";
			break;
			
		    case CUFFT_UNALIGNED_DATA:
			error_msg <<  "CUFFT_UNALIGNED_DATA";
			break;
			
		    default:
			error_msg <<  "<unknown: "<<error<<">";
		    }
		error_msg << std::endl;
		throw std::runtime_error(error_msg.str());
	    }
	}
	
	inline
	void check_file_error(std::ifstream& infile, std::string filename){
	    if(!infile.good()) {
		std::stringstream error_msg;
		error_msg << "File "<< filename << " could not be opened: ";
		if ( (infile.rdstate() & std::ifstream::failbit ) != 0 )
		    error_msg << "Logical error on i/o operation" << std::endl;
		if ( (infile.rdstate() & std::ifstream::badbit ) != 0 )
		    error_msg << "Read/writing error on i/o operation" << std::endl;
		if ( (infile.rdstate() & std::ifstream::eofbit ) != 0 )
		    error_msg << "End-of-File reached on input operation" << std::endl;
		throw std::runtime_error(error_msg.str());
	    }
	}
	
	inline
	void check_file_error(std::ofstream& infile, std::string filename){
	    if(!infile.good()) {
		std::stringstream error_msg;
		error_msg << "File "<< filename << " could not be opened: ";
		if ( (infile.rdstate() & std::ifstream::failbit ) != 0 )
		    error_msg << "Logical error on i/o operation" << std::endl;
		if ( (infile.rdstate() & std::ifstream::badbit ) != 0 )
		    error_msg << "Read/writing error on i/o operation" << std::endl;
		if ( (infile.rdstate() & std::ifstream::eofbit ) != 0 )
		    error_msg << "End-of-File reached on input operation" << std::endl;
		throw std::runtime_error(error_msg.str());
	    }
	}

	template <class VectorType>
	void write_vector(VectorType& out, std::string filename){
	    std::ofstream FILE(filename, std::ios::out | std::ofstream::binary);
	    check_file_error(FILE,filename);
	    FILE.write(reinterpret_cast<char*>(&out[0]), out.size()*sizeof(typename VectorType::value_type)); 
	    FILE.close();
	}
	
	/*
	template <class VectorType>
        void read_vector(VectorType& in, std::string filename){
	    std::ifstream INFILE(filename, std::ios::in | std::ifstream::binary);
	    std::istreambuf_iterator<typename VectorType::value_type> iter(INFILE);
	    std::copy(iter.begin(), iter.end(), std::back_inserter(in));
	    }*/

	int gpu_count(){
	    int count;
	    cudaGetDeviceCount(&count);
	    return count;
	}
    
	void print_stack_trace(unsigned int max_depth){
	    int trace_depth;
	    void *buffer[max_depth];
	    char **strings;
	    trace_depth = backtrace(buffer, max_depth);
	    strings = backtrace_symbols(buffer, trace_depth);
	    if (strings == NULL) {
		std::cerr << "Stack trace failed" << std::endl;
	    } else {
		for (int jj = 0; jj < trace_depth; jj++)
		    std::cerr << strings[jj] << std::endl;
		free(strings);
	    }
	}
    } /* namespace utils */ 
} // namespace peasoup

#endif
