#include <sstream>
#include <thread>
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
	    LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
                "Preparing PeakFinder\n",
                "Fundamental metadata:\n",fundamental.metadata.display(),
                "Fundamental size: ",fundamental.data.size()," samples\n",
		"Harmonics metadata:\n",harmonics.metadata.display(),
                "Harmonics size: ",harmonics.data.size()," samples");
	    size_t size = fundamental.data.size();
	    idxs.resize(size);
	    powers.resize(size);
	    h_idxs.resize(size);
	    h_powers.resize(size);
	    int nh = harmonics.metadata.binwidths.size();
	    bool nn = fundamental.metadata.nn;
	    thresholds.push_back(cand_utils::power_for_sigma(minsigma, 1, (float) 1.0, nn));
	    for (int ii=0;ii<nh;ii++)
		thresholds.push_back(cand_utils::power_for_sigma(minsigma, 1<<(ii+1), (float) 1.0,nn));
	    std::stringstream thresholds_str;
	    for (int jj=0; jj<thresholds.size(); jj++)
		thresholds_str << "Nh=" << jj << ": " << thresholds[jj] << "\n";
	    LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
		"\nThreshold powers:\n",thresholds_str.str());
	}
	
	template <System system, typename T>
        int PeakFinder<system,T>::_execute(pow_iter in, size_t size, float thresh, int offset)
        {
            using thrust::tuple;
            using thrust::counting_iterator;
            using thrust::zip_iterator;
            counting_iterator<unsigned> index_counter(offset);
            zip_iterator< peak_tuple_in > input_zip = make_zip_iterator(make_tuple(index_counter,in));
            zip_iterator< peak_tuple_out > output_zip = make_zip_iterator(make_tuple(idxs.begin(),powers.begin()));
	    int num_copied = thrust::copy_if(this->get_policy(), input_zip, input_zip+size, output_zip,
					     functor::greater_than_threshold<float>(thresh)) - output_zip;
	    return num_copied;
	}

	    
	template <System system, typename T>
        void PeakFinder<system,T>::execute()
	{
	    int num_copied,max_bin,min_bin;
	    size_t size = fundamental.data.size();
	    pow_iter in = fundamental.data.begin();
	    max_bin = max_freq/fundamental.metadata.binwidth;
            min_bin = min_freq/fundamental.metadata.binwidth;
	    LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
                "Executing PeakFinder on Nh = ",0);
	    num_copied = _execute(in+min_bin,max_bin-min_bin,thresholds[0],min_bin);
	    filter_unique(num_copied,fundamental.metadata.binwidth,0);
	    in = harmonics.data.begin();
	    std::vector<float>& binwidths = harmonics.metadata.binwidths;
	    for (int ii=0; ii<binwidths.size(); ii++){
		LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
		    "Executing PeakFinder on Nh = ",ii+1);
		int offset = ii*size;
		max_bin = std::min((size_t)(max_freq/binwidths[ii]),size);
		min_bin = min_freq/binwidths[ii];
		num_copied = _execute(in+offset+min_bin,max_bin-min_bin,thresholds[ii+1],min_bin);
		filter_unique(num_copied,binwidths[ii],ii+1);
	    }
	}
	
	template <System system, typename T>
        void PeakFinder<system,T>::filter_unique(int num_copied,float df,int nh)
	{
	    if (num_copied<1){
		LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
                    "No candidates returned for DM = ",fundamental.metadata.dm,
		    " pccm, acc = ",fundamental.metadata.acc," m/s/s");
		return;
	    }
	    LOG(logging::get_logger("transform.peakfinder"),logging::DEBUG,
		"Filtering unique candidates");

	    thrust::copy(idxs.begin(),idxs.begin()+num_copied,h_idxs.begin());
	    thrust::copy(powers.begin(),powers.begin()+num_copied,h_powers.begin());
	    this->sync();

	    size_t new_size = dets.size()+num_copied;
            if (dets.capacity() < new_size)
                dets.reserve(2 * new_size);
	    
	    int ii = 0;
	    float cpeak = h_powers[ii];
	    int cpeakidx = h_idxs[ii];
	    ii++;
	    auto& info = fundamental.metadata;
	    while(ii < num_copied){
		if ((h_idxs[ii]-h_idxs[ii-1]) > 1){
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
		ii++;
	    }
	    dets.push_back(type::Detection(df*cpeakidx,cpeak,nh,info.acc,info.dm));
	    double dsize = (double) fundamental.data.size();
	    for (auto& det: dets)
		det.sigma = cand_utils::candidate_sigma(det.power,1<<det.nh,1,fundamental.metadata.nn);
	}
    } //transform
} //peasoup


