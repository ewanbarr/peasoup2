#include <utility>

#include "thrust/for_each.h"
#include "transforms/zapper.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T> 
	    zapper_functor<T>::zapper_functor(thrust::complex<T>* ar):ar(ar){}
	    
	    template <typename T>
	    inline __host__ __device__ 
	    void zapper_functor<T>::operator()(unsigned bin)
	    {
		ar[bin] = thrust::complex<T>(0.0,0.0);
	    }
	    
	} //namespace functor

	template <System system, typename T>
	void Zapper<system,T>::prepare()
	{
	    LOG(logging::get_logger("transform.zapper"),logging::DEBUG,
                "Preparing Zapper\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples\n",
		"Nbirds: ",birdies.size());
	    unsigned lower,upper;
	    float df = input.metadata.binwidth;
	    unsigned size = input.data.size();
	    for (auto i: birdies) {
		float freq = std::get<0>(i);
		float width = std::get<1>(i);
		int bin = (freq/df + 0.5);
		int binw = (width/df);
		lower = std::max(bin-binw,0);
		upper =std::min(bin+binw,(int)size);
		for (int ii=lower;ii<upper;ii++)
		    bins.push_back(ii);
	    }
	}
	
	template <System system, typename T>
	void Zapper<system,T>::execute()
	{
	    if (bins.size()==0)
		return;
	    else {
		LOG(logging::get_logger("transform.zapper"),logging::DEBUG,"Executing birdie zapper");
		thrust::complex<T>* ar = thrust::raw_pointer_cast(input.data.data());
		thrust::for_each(this->get_policy(),bins.begin(),bins.end(),
				 functor::zapper_functor<T>(ar));
	    }
	}

    } // namespace transform
} // namespace peasoup
