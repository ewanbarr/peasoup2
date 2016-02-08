#ifndef PEASOUP_TRANSFORM_BASE_CUH
#define PEASOUP_TRANSFORM_BASE_CUH

#include <cstddef>
#include "cuda.h"
#include <thrust/execution_policy.h>
#include "misc/system.cuh"
#include "misc/policies.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace transform {
	
	using namespace thrust::detail;
	using namespace thrust::system::cuda::detail;

	class TransformBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void execute()=0;
	    virtual void set_stream(cudaStream_t stream)=0;
	    virtual void sync()=0;
	};

	template <System system>
	class Transform: public TransformBase
	{
        };

	template <>
	class Transform<HOST>: public TransformBase
	{
	public:
	    host_t get_policy(){return thrust::host;}
	    void set_stream(cudaStream_t stream){}
	    void sync(){}
	    Transform(){}
	};

	template <>
        class Transform<DEVICE>: public TransformBase
        {
        protected:
	    typedef execute_with_allocator<policy::cached_allocator, execute_on_stream_base > policy_type;
	    cudaStream_t stream;
	    policy::cached_allocator allocator;
	    
	public:
	    void set_stream(cudaStream_t stream){this->stream=stream;}
	    void sync(){
		if (stream == nullptr)
		    cudaDeviceSynchronize();
                else
                    cudaStreamSynchronize(stream);
	    }
	    policy_type get_policy(){
		if (stream == nullptr)
		    return thrust::cuda::par(allocator);
		else
		    return thrust::cuda::par(allocator).on(stream);
	    }
	    Transform():stream(nullptr){}
	};
	

    } // transform
} // peasoup

#endif
