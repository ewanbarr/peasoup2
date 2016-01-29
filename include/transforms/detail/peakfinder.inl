#include "utils/chi2lib.hpp"
#include "transforms/peakfinder.cuh"

namespace peasoup {
    namespace transform {
	namespace functor {

	    template <typename T>
	    inline __host__ __device__ 
	    bool greater_than_threshold<T>::operator()(thrust::tuple<unsigned,T> t) const 
	    { return thrust::get<1>(t) > threshold;  }

	} // namespace functor

	template <System system, typename T>
	std::vector<float>& PeakFinder<system,T>::get_thresholds()
	{
	    return thresholds;
	}

	template <System system, typename T>
	void PeakFinder<system,T>::prepare()
	{
	    utils::print(__PRETTY_FUNCTION__,"\n");
            fundamental.metadata.display();
	    harmonics.metadata.display();
	    size_t size = fundamental.data.size();
	    idxs.resize(size);
	    powers.resize(size);
	    h_idxs.resize(size);
	    h_powers.resize(size);
	    int nh = harmonics.metadata.binwidths.size();
	    bool nn = fundamental.metadata.nn;
	    thresholds.push_back(cand_utils::power_for_sigma(minsigma, 1, (float) size, nn));
	    for (int ii=0;ii<nh;ii++)
		thresholds.push_back(cand_utils::power_for_sigma(minsigma, 1<<(ii+1), (float) size, nn));
	}
	
	template <System system, typename T>
        void PeakFinder<system,T>::_execute(pow_iter in, size_t size, int nh, float df, float thresh)
	{
	    using thrust::tuple;
	    using thrust::counting_iterator;
	    using thrust::zip_iterator;
	    counting_iterator<unsigned> index_counter(0);
	    zip_iterator< peak_tuple_in > input_zip = make_zip_iterator(make_tuple(index_counter,in));
	    zip_iterator< peak_tuple_out > output_zip = make_zip_iterator(make_tuple(idxs.begin(),powers.begin()));
	    int num_copied = thrust::copy_if(this->get_policy(), input_zip, input_zip+size, output_zip,
					     functor::greater_than_threshold<float>(thresh)) - output_zip;
	    thrust::copy(idxs.begin(),idxs.begin()+num_copied,h_idxs.begin());
	    thrust::copy(powers.begin(),powers.begin()+num_copied,h_powers.begin());
	    dets.reserve(dets.size()+num_copied);
	    filter_unique(num_copied,df,nh);
	}
	    
	template <System system, typename T>
        void PeakFinder<system,T>::execute()
	{
	    size_t size = fundamental.data.size();
	    pow_iter in = fundamental.data.begin();
	    _execute(in,size,0,fundamental.metadata.binwidth,thresholds[0]);
	    in = harmonics.data.begin();
	    std::vector<float>& binwidths = harmonics.metadata.binwidths;
	    for (int ii=0; ii<binwidths.size(); ii++){
		int offset = ii*size;
		_execute(in+offset,size,ii+1,binwidths[ii],thresholds[ii+1]);
	    }
	}

	template <System system, typename T>
        void PeakFinder<system,T>::filter_unique(int num_copied,float df,int nh)
	{
	    if (num_copied<1) return;
	    int ii = 0;
	    float cpeak = h_powers[ii];
	    int cpeakidx = h_idxs[ii];
	    int lastidx = cpeakidx;
	    auto& info = fundamental.metadata;
	    while(ii < num_copied){
		ii++;
		if ((h_idxs[ii]-lastidx) > 1){
		    dets.push_back(type::Detection(df*cpeakidx,cpeak,nh,info.acc,info.dm));
		    cpeak = h_powers[ii];
		    cpeakidx = h_idxs[ii];
		    ii++;
		} else {
		    if (h_powers[ii] > cpeak){
			cpeak = h_powers[ii];
			cpeakidx = h_idxs[ii];
		    }
		}
	    }
	    dets.push_back(type::Detection(df*cpeakidx,cpeak,nh,info.acc,info.dm));
	}
    } //transform
} //peasoup


