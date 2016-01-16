#ifndef PEASOUP_ZAPPER_CUH
#define PEASOUP_ZAPPER_CUH

#include <vector>
#include <utility>

#include "thrust/complex.h"
#include "misc/system.cuh"
#include "data_types/frequencyseries.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {

	    template <typename T>
	    struct zapper_functor
	    {
		thrust::complex<T>* ar;
		zapper_functor(thrust::complex<T>* ar);
		inline __host__ __device__ void operator()(unsigned bin);
	    };

	} //namespace functor

	class ZapperBase 
	{
	public:
	    virtual void prepare()=0;
	    virtual void execute()=0;
	};

	template <System system, typename T>
	class Zapper: public ZapperBase
	{
	private:
	    type::FrequencySeries<system,thrust::complex<T> >& input;
	    std::vector<std::pair<float,float> >& birdies;
	    typename SystemVector<system,unsigned>::vector_type bins;
	    SystemPolicy<system> policy_traits;
	    
	public:
	    Zapper(type::FrequencySeries<system,thrust::complex<T> >& input,
		   std::vector<std::pair<float,float> >& birdies)
		:input(input),birdies(birdies) {}
	    void prepare();	    
	    void execute();
	};
	
    } // namespace transform
} // namspace peasoup

#include "transforms/detail/zapper.inl"

#endif //PEASOUP_ZAPPER_CUH
