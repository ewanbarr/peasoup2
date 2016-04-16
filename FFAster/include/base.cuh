
#ifndef FFASTER_BASE_H_
#define FFASTER_BASE_H_

#include "ffaster.h"

namespace FFAster
{
  namespace Base
  {
    
    class TempMemoryStore
    {
    protected:
      void * tmp_storage;
      size_t tmp_storage_bytes;

    public:
      TempMemoryStore()
	:tmp_storage(NULL),
	 tmp_storage_bytes(0){}
      
      virtual void set_tmp_storage_buffer(void* block, 
					  size_t nbytes)
      {
	this->tmp_storage = block;
	this->tmp_storage_bytes = nbytes;
      }
      virtual size_t get_tmp_storage_bytes(){ return this->tmp_storage_bytes;}
      virtual void * get_tmp_storage_buffer(){ return this->tmp_storage;}
    };
    
    class Transform: public TempMemoryStore
    {
    public:
      virtual size_t get_required_tmp_bytes(ffa_params_t& plan){return 0;}
      virtual size_t get_required_output_bytes(ffa_params_t& plan){return 0;}
    };
    
    class HostTransform: public Transform
    {
    };
    
    class DeviceTransform: public Transform
    {
    protected:
      cudaStream_t stream;
    public:
      DeviceTransform()
        :stream(0){}
      virtual void set_stream(cudaStream_t stream){this->stream = stream;}
      cudaStream_t get_stream(){return stream;}
    };
    
  };
};

#endif
