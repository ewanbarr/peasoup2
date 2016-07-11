#ifndef FFASTER_UTILS_CUH_
#define FFASTER_UTILS_CUH_

#include "cuda.h"
#include "cufft.h"
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <execinfo.h>
#include "cuda.h"
#include "cuda_runtime.h"

namespace FFAster {
  namespace Utils {
    
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
    
    template <class T>
    void device_malloc(T** ptr,unsigned int units){
      cudaError_t error = cudaMalloc((void**)ptr, sizeof(T)*units);
      check_cuda_error("Error from device_malloc");
    }

    template <class T>
    void host_malloc(T** ptr,unsigned int units){
      cudaMallocHost((void**)ptr, sizeof(T)*units);
      check_cuda_error("Error from host_malloc");
    }
    
    template <class T>
    void device_free(T* ptr){
      cudaFree(ptr);
      check_cuda_error("Error from device_free");
    }
    
    template <class T>
    void host_free(T* ptr){
      cudaFreeHost((void*) ptr);
      check_cuda_error("Error from host_free.");
    }
    
    template <class T>
    void h2dcpy(T* d_ptr, T* h_ptr, unsigned int units){
      cudaMemcpy(d_ptr,h_ptr,sizeof(T)*units,cudaMemcpyHostToDevice);
      check_cuda_error("Error from h2dcpy");
    }
    
    template <class T>
    void d2hcpy(T* h_ptr, T* d_ptr, unsigned int units){
      cudaMemcpy(h_ptr,d_ptr,sizeof(T)*units,cudaMemcpyDeviceToHost);
      check_cuda_error("Error from d2hcpy");
    }
    
    template <class T>
    void d2dcpy(T* d_ptr_dst, T* d_ptr_src, unsigned int units){
      cudaMemcpy(d_ptr_dst,d_ptr_src,sizeof(T)*units,cudaMemcpyDeviceToDevice);
      check_cuda_error("Error from d2dcpy");
    }

    template <class T>
    void h2hcpy(T* h_ptr_dst, const T* h_ptr_src, unsigned int units){
      if (memcpy(h_ptr_dst,h_ptr_src,sizeof(T)*units) == NULL)
	throw std::runtime_error("memcpy failed");
    }
    
    template <class T>
    void dump_device_buffer(T* buffer, size_t size, std::string filename){
      T* host_ptr;
      host_malloc<T>(&host_ptr,size);
      d2hcpy(host_ptr,buffer,size);
      std::ofstream infile;
      infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
      infile.write((char*)host_ptr ,size*sizeof(T));
      infile.close();
      host_free(host_ptr);
    }

    template <class T>
    void dump_host_buffer(T* buffer, size_t size, std::string filename){
      std::ofstream infile;
      infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
      infile.write((char*)buffer ,size*sizeof(T));
      infile.close();
    }

    inline
    int gpu_count(){
      int count;
      cudaGetDeviceCount(&count);
      return count;
    }
    
    inline
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

  } /* namespace Utils */ 
  
  
  namespace Allocators
  {
    
    class ScratchAllocator
    {
    protected:
      void * slab;
      size_t total_bytes;
      char * ptr;
      size_t byte;
      
    public:
      ScratchAllocator(void* slab, size_t total_bytes):
	slab(slab),
        total_bytes(total_bytes),
        byte(0)
      {
	ptr = (char*) slab;
      }

      template<class T>
      void device_allocate(T** data, size_t nunits)
      {
        if (byte+nunits*sizeof(T) > total_bytes)
          throw std::runtime_error("request exceeds slab size");
        *data = (T*) ptr;
        byte += nunits*sizeof(T);
        ptr = (char*)slab+byte;
      }

      template<class T>
      void device_free(T *data){}
    };
    
    class SlabAllocator: public ScratchAllocator
    {
    public:
      SlabAllocator(size_t total_bytes)
	:ScratchAllocator(NULL,total_bytes)
      {
	//printf("Allocating %zu bytes\n",total_bytes);
	cudaMalloc((void**)&slab, total_bytes);
	Utils::check_cuda_error("Error from SlabAllocator::cudaMalloc");
	ptr = (char*) slab;
      }

      ~SlabAllocator()
      {
	cudaFree(slab);
	Utils::check_cuda_error("Error from SlabAllocator::cudaFree");
      }
    };

    

    /*
    class BaseAllocator
    {
    public:
      BaseAllocator(){};
      ~BaseAllocator(){};
      
      virtual void device_allocate(void* &data, size_t nunits) {}
      virtual void device_free(void* data) {}
    };
    
    
    class SimpleAllocator: public BaseAllocator
    {
    public:
      SimpleAllocator(){};
      ~SimpleAllocator(){};

      template <class T>
      void device_allocate(T* &data, size_t nunits) { Utils::device_malloc<T>(&data,(unsigned int) nunints); }

      template <class T>
      void device_free(T* data) { Utils::device_free(data); }
    };

    
    class CubAllocator: public BaseAllocator
    {
    private:
      cub::CachingDeviceAllocator allocator;
      
    public:
      CubAllocator(){};
      ~CubAllocator(){};
      
      template<class T>
      void device_allocate(T *&data, size_t nunits)
      {
	allocator.DeviceAllocate((void**)&data,nunints*sizeof(T));
	Utils::check_cuda_error("Error from CubAllocator::device_allocate")
      }
      
      template <class T>
      void device_free(T* data) 
      {
	allocator.DeviceFree((void*)data);
	Utils::check_cuda_error("Error from CubAllocator::device_free")
      }
    };
    
    
    class SlabAllocator: public BaseAllocator
    {
    private:
      void * slab;
      size_t total_bytes;
      void * ptr;
      
    public:
      SlabAllocator(size_t total_bytes):
	total_bytes(total_bytes)
      {
	Utils::device_malloc<void>(&slab,(unsigned int)total_bytes);
      }

      ~SlabAllocator()
      {
	Utils::device_free<void>(slab);
      }
      
      template<class T>
      void device_allocate(T* &data, size_t nunits)
      {
	if (ptr+nunits*sizeof(T) >= slab+total_bytes)
	  throw std::runtime_error("request exceeds slab size");
	data = ptr;
	ptr += nunits*sizeof(T);
      }
      
      template<class T>
      void device_free(T *data){}
      };*/
      
  } 
  
} /* namespace FFAster */

#endif
