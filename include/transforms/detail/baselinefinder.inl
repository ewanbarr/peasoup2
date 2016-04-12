#include "transforms/baselinefinder.cuh"
#include <sstream>
#include <cmath>

namespace peasoup {
    namespace transform {
	template <typename T>
	inline __host__ __device__
	T median3(T a, T b, T c) {
	    return a < b ? b < c ? b
		: a < c ? c : a
		: a < c ? a
		: b < c ? c : b;
	}
	
	template <typename T>
	inline __host__ __device__
	T median4(T a, T b, T c, T d) {
	    return a < c ? b < d ? a < b ? c < d ? 0.5f*(b+c) : 0.5f*(b+d)
		: c < d ? 0.5f*(a+c) : 0.5f*(a+d)
		: a < d ? c < b ? 0.5f*(d+c) : 0.5f*(b+d)
		: c < b ? 0.5f*(a+c) : 0.5f*(a+b)
		: b < d ? c < b ? a < d ? 0.5f*(b+a) : 0.5f*(b+d)
		: a < d ? 0.5f*(a+c) : 0.5f*(c+d)
		: c < d ? a < b ? 0.5f*(d+a) : 0.5f*(b+d)
		: a < b ? 0.5f*(a+c) : 0.5f*(c+b);
	}
	
	template <typename T>
	inline __host__ __device__
	T median5(T a, T b, T c, T d, T e) {
	    return b < a ? d < c ? b < d ? a < e ? a < d ? e < d ? e : d
		: c < a ? c : a
		: e < d ? a < d ? a : d
		: c < e ? c : e
		: c < e ? b < c ? a < c ? a : c
		: e < b ? e : b
		: b < e ? a < e ? a : e
		: c < b ? c : b
		: b < c ? a < e ? a < c ? e < c ? e : c
		: d < a ? d : a
		: e < c ? a < c ? a : c
		: d < e ? d : e
		: d < e ? b < d ? a < d ? a : d
		: e < b ? e : b
		: b < e ? a < e ? a : e
		: d < b ? d : b
		: d < c ? a < d ? b < e ? b < d ? e < d ? e : d
		: c < b ? c : b
		: e < d ? b < d ? b : d
		: c < e ? c : e
		: c < e ? a < c ? b < c ? b : c
		: e < a ? e : a
		: a < e ? b < e ? b : e
		: c < a ? c : a
		: a < c ? b < e ? b < c ? e < c ? e : c
		: d < b ? d : b
		: e < c ? b < c ? b : c
		: d < e ? d : e
		: d < e ? a < d ? b < d ? b : d
		: e < a ? e : a
		: a < e ? b < e ? b : e
		: d < a ? d : a;
	}
	
	namespace functor {
	    template <typename T> 
	    inline __host__ __device__
	    T median5_functor<T>::operator()(unsigned int i) const
	    {
		T a = in[5*i+0];
		T b = in[5*i+1];
		T c = in[5*i+2];
		T d = in[5*i+3];
		T e = in[5*i+4];
		return median5<T>(a, b, c, d, e);
	    }
	    	    
	    template <typename T>
	    inline __host__ __device__
	    T linear_stretch_functor<T>::operator()(unsigned out_idx) const
	    {
		float fidx = ((float)out_idx) / step - correction;
		unsigned idx = (unsigned) fidx;
		if (fidx<0)
		    idx = 0;
		else if (idx + 1 >= in_size)
		    idx = in_size-2;
		return in[idx] + ((in[idx+1] - in[idx]) * (fidx-idx));
	    }
	} // namespace functor
	
	
	template <System system, typename T>
	void BaselineFinder<system,T>::median_scrunch5(const vector_type& in, vector_type& out)
	{
	    size_t count = in.size();
	    const T* ptr = thrust::raw_pointer_cast(in.data());
	    if( count == 1 ) 
		out[0] = in[0];
	    else if( count == 2 ) 
		out[0] = 0.5f*(in[0] + in[1]);
	    else if( count == 3 ) 
		out[0] = median3<T>(in[0],in[1],in[2]);
	    else if( count == 4 ) 
		out[0] = median4<T>(in[0],in[1],in[2],in[3]);
	    else {
		thrust::transform(this->get_policy(),
				  thrust::make_counting_iterator<unsigned int>(0),
				  thrust::make_counting_iterator<unsigned int>(in.size()/5),
				  out.begin(), functor::median5_functor<T>(ptr));
	    }
	}
	 
	template <System system, typename T>
	void BaselineFinder<system,T>::linear_stretch(const vector_type& in, vector_type& out, float step)
	{
	    const T* ptr = thrust::raw_pointer_cast(in.data());
	    thrust::transform(this->get_policy(),
			      thrust::make_counting_iterator<unsigned int>(0),
			      thrust::make_counting_iterator<unsigned int>(out.size()),
			      out.begin(), functor::linear_stretch_functor<T>(ptr, in.size(), step));
	}
	
	template <System system, typename T>
	void FDBaselineFinder<system,T>::prepare()
	{
	    LOG(logging::get_logger("transform.fdbaselinefinder"),logging::DEBUG,
                "Preparing FDBaselineFinder\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    
	    output.data.resize(input.data.size());
	    intermediate.resize(input.data.size());
	    output.metadata = input.metadata;
	    boundaries.clear();
	    float tobs = 1/input.metadata.binwidth;
	    float nyquist = input.metadata.binwidth * input.data.size();
	    unsigned k = 10;
	    unsigned power = 1;
	    unsigned window;
	    size_t boundary_idx;
	    float boundary;
	    
	    std::stringstream tmp;
	    while (true) {
		window = pow((unsigned)5,(unsigned)power) ;
		boundary = SPEED_OF_LIGHT * window / (k * accel_max * tobs * tobs);
		boundary_idx = min((unsigned) (boundary/input.metadata.binwidth), (unsigned) input.data.size());
		boundaries.push_back(boundary_idx);
		tmp << "Boundary: "<<boundary<<" Hz ("<<boundary_idx<<") power "<<window<<"\n";
		this->medians.push_back( vector_type(input.data.size()/window) );
		if (boundary>nyquist)
		    break;
		power+=1;
	    }
	    LOG(logging::get_logger("transform.fdbaselinefinder"),logging::DEBUG,
		"\n Median smoothing window boundaries:\n",tmp.str());
	    
	    LOG(logging::get_logger("transform.fdbaselinefinder"),logging::DEBUG,
		"Prepared FDBaselineFinder\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
	}
	
	template <System system, typename T>
        void FDBaselineFinder<system,T>::execute()
	{
	    LOG(logging::get_logger("transform.fdbaselinefinder"),logging::DEBUG,
                "Executing baseline finder");
	    vector_type* in = &(input.data);
	    vector_type* out;
	    size_t offset = 0;
	    float step = 5;
	    for (int ii=0;ii<boundaries.size();ii++){
		out = &(this->medians)[ii];
		this->median_scrunch5(*in,*out);
		this->linear_stretch(*out,intermediate,step);
		thrust::copy_n(intermediate.begin()+offset,
			       boundaries[ii]-offset,
			       output.data.begin()+offset);
		offset = boundaries[ii];
		in = out;
		step *= 5;
	    }
	}

	template <System system, typename T>
	void TDBaselineFinder<system,T>::prepare()
        {
            LOG(logging::get_logger("transform.tdbaselinefinder"),logging::DEBUG,
                "Preparing TDBaselineFinder\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");

            output.data.resize(input.data.size());
	    output.metadata = input.metadata;
	    
	    float window = smoothing_interval/input.metadata.tsamp;
	    unsigned nsteps = (unsigned) (std::log10(window)/std::log10(5.0));
	    unsigned step = 5;
	    for (int ii=0;ii<nsteps;ii++)
		{
		    this->medians.push_back( vector_type(input.data.size()/step) );
		    step*=5;
		}
            LOG(logging::get_logger("transform.tdbaselinefinder"),logging::DEBUG,
                "Prepared TDBaselineFinder\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
        }
	
	template <System system, typename T>
        void TDBaselineFinder<system,T>::execute()
        {
            LOG(logging::get_logger("transform.tdbaselinefinder"),logging::DEBUG,
                "Executing baseline finder");
            vector_type* in = &(input.data);
            vector_type* out;
            float step = 5;
            for (int ii=0;ii<this->medians.size();ii++){
                out = &(this->medians)[ii];
                this->median_scrunch5(*in,*out);
		in = out;
		step*=5;
	    }
	    this->linear_stretch(*out,output.data,step/5);
        }



    } //transform
} //peasoup



