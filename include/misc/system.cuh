#ifndef PEASOUP_SYSTEM_CUH
#define PEASOUP_SYSTEM_CUH

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/system/cuda/experimental/pinned_allocator.h"

namespace peasoup {
    
    enum System {
	DEVICE,
	HOST
    };
    
    /* Trait structs to define container types */
    template <System system, typename T> struct SystemVector {};
    
    template <typename T>
    struct SystemVector<DEVICE,T>
    { 
	typedef thrust::device_vector<T> vector_type; 
	typedef thrust::device_ptr<T> ptr_type;
    };
    
    template <typename T>
    struct SystemVector<HOST,T>
    { 
	typedef thrust::host_vector<T,thrust::cuda::experimental::pinned_allocator<T> > vector_type; 
	typedef T* ptr_type;
    };

    
    /* Trait structs to define execution policies */
    template <System system> struct SystemPolicy {};
    
    template <>
    struct SystemPolicy<DEVICE>
    { 
	const thrust::detail::device_t policy; 
	SystemPolicy():policy(thrust::device){};
    };
    
    template <>
    struct SystemPolicy<HOST>
    { 
	const thrust::detail::host_t policy; 
	SystemPolicy():policy(thrust::host){};
    };
    
    //const thrust::detail::device_t SystemPolicy<DEVICE>::policy = thrust::device;
    //const thrust::detail::host_t SystemPolicy<HOST>::policy = thrust::host;
    
} // namespace peasoup

#endif // PEASOUP_SYSTEM_CUH
