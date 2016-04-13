#include <cstddef>
#include <stdexcept>
#include <map>
#include "transforms/downsampler.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    inline __host__ __device__
	    T downsample<T>::operator()(unsigned int idx)
	    {
		 int start = idx*factor;
		 int end = start+factor;
		 end = end < size ? end : size;
		 T sum = 0;
		 for (int ii=start;ii<end;ii++){
		     sum += input[ii];
		 }
		 return sum/(end-start);
	     }

	} //functor
	
	template <System system, typename T>
	CachedDownsampler<system,T>::CachedDownsampler(type::TimeSeries<system,T>* data,
						       unsigned int max_factor)
	    :data(data),
	     max_factor(max_factor),
	     downsampled_factor(1),
	     parent(nullptr),
	     new_parent(false)
	{
	    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
                "Building cached downsampler head node\n");
	    factoriser = new utils::Factoriser;
	}
	
	template <System system, typename T>
	CachedDownsampler<system,T>::CachedDownsampler(CachedDownsampler<system,T>* parent,
						       type::TimeSeries<system,T>* data,
						       unsigned int downsampled_factor)
	    :parent(parent),
	     data(data),
	     downsampled_factor(downsampled_factor),
	     new_parent(false)
	{
	    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
                "Building cached downsampler child node\n",
		"Downsampling factor: ",downsampled_factor,"\n");
	    factoriser = parent->factoriser;
	    max_factor = parent->max_factor;
	}

	template <System system, typename T>
	CachedDownsampler<system,T>::~CachedDownsampler()
	{
	    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
                "Cleaning up cached downsampler instance\n");
	    typedef typename std::map<unsigned int, CachedDownsampler<system,T>* >::iterator it_type;
	    for(it_type iterator = cache.begin(); iterator != cache.end(); iterator++)
		delete iterator->second;
	    if (parent != nullptr)
		delete data;
	    else
		delete factoriser;
	}

	template <System system, typename T>
        void CachedDownsampler<system,T>::set_data(type::TimeSeries<system,T>* data)
	{
	    this->data = data;
	    this->notify();
	}
	
	template <System system, typename T>
        void CachedDownsampler<system,T>::notify()
	{
	    typedef typename std::map<unsigned int, CachedDownsampler<system,T>* >::iterator it_type;
	    for(it_type iterator = cache.begin(); iterator != cache.end(); iterator++){
		iterator->second->new_parent = true;
		iterator->second->notify();
	    }
	}

	template <System system, typename T>
	unsigned int CachedDownsampler<system,T>::closest_factor(unsigned int factor)
	{
	    return factoriser->get_nearest_factor(factor,max_factor);
	}
	
	template <System system, typename T>
	CachedDownsampler<system,T>* CachedDownsampler<system,T>::downsample(unsigned int factor)
	{
	    	    
	    if (factor<1){
		LOG(logging::get_logger("transform.downsampler"),logging::ERROR,
		    "Bad downsampling factor passed: ",factor,"\n");
		throw std::runtime_error("Downsampling factor must be greater than zero");
	    }
	    
	    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
                "Downsampling by a factor of ",factor,"\n");
	    if (factor==1){
		return this;
	    }
	    else {
		unsigned int first_factor = factoriser->first_factor(factor);
		if (cache.count(first_factor)){
		    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
			"Found cached downsampling\n");
		    if (cache[first_factor]->new_parent){
			LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
			    "New data in parent, redoing downsampling\n");
			size_t downsampled_size = data->data.size()/first_factor;
			type::TimeSeries<system,T>* downsampled_data = cache[first_factor]->data;
			thrust::counting_iterator<unsigned> begin(0);
			thrust::counting_iterator<unsigned> end = begin + downsampled_size;
			thrust::transform(SystemPolicy<system>().policy,begin,end,
					  downsampled_data->data.data(),
					  functor::downsample<T>(data->data.data(),
								 data->data.size(),first_factor));
			cache[first_factor]->new_parent = false;
		    }
		    return cache[first_factor]->downsample(factor/first_factor);
		}
		else {
		    LOG(logging::get_logger("transform.downsampler"),logging::DEBUG,
			"No cached instance, partial downsampling by ",first_factor,"\n");
		    size_t downsampled_size = data->data.size()/first_factor;
		    type::TimeSeries<system,T>* downsampled_data = new type::TimeSeries<system,T>;
		    downsampled_data->data.resize(downsampled_size);
		    downsampled_data->metadata = data->metadata;
		    downsampled_data->metadata.tsamp*=first_factor;
		    thrust::counting_iterator<unsigned> begin(0);
		    thrust::counting_iterator<unsigned> end = begin + downsampled_size;
		    thrust::transform(SystemPolicy<system>().policy,begin,end,
				      downsampled_data->data.data(),
				     functor::downsample<T>(data->data.data(),
							    data->data.size(),first_factor));
		    cache[first_factor] = new CachedDownsampler<system,T>
			(this,downsampled_data,downsampled_factor*first_factor);
		    return cache[first_factor]->downsample(factor/first_factor);
		}
	    }
	}
	
    } // transform
} // peasoup

