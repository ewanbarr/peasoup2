#include <vector>

namespace peasoup {
    namespace pipeline {
	inline void StaticAccelerationPlan::get_accelerations(std::vector<float>& acc_list, float dm)
	{
	    acc_list = list;
	}
	
	inline DMDependentAccelerationPlan::DMDependentAccelerationPlan(float acc_start, 
									float acc_end, 
									float acc_tol, 
									float acc_pulse_width, 
									size_t nsamps,
									float tsamp, 
									float cfreq, 
									float bw)
	    :acc_start(acc_start), 
	     acc_end(acc_end), 
	     acc_tol(acc_tol), 
	     acc_pulse_width(acc_pulse_width),
	     nsamps(nsamps), 
	     tsamp(tsamp),
	     cfreq(cfreq),
	     bw(fabs(bw))
	    {
		tsamp_us = 1.0e6 * tsamp;
		tobs = nsamps*tsamp;
		cfreq_GHz = 1.0e-3 * cfreq;
		acc_pulse_width /= 1.0e3;
	    }
	    
	inline void DMDependentAccelerationPlan::get_accelerations(std::vector<float>& acc_list, float dm)
	{
	    acc_list.clear();
	    if (acc_end==acc_start){
		acc_list.push_back(0.0);
		return;
	    }
	    
	    //channel smearing
	    float tdm = pow(8.3*bw/pow(cfreq,3.0)*dm,2.0);
	    //intrinsic smearing
	    float tpulse = acc_pulse_width * acc_pulse_width;
	    //sampling smearing
	    float ttsamp = tsamp * tsamp;
	    //observed width
	    float w_us = sqrt(tdm+tpulse+ttsamp);
	    float alt_a = 2.0 * w_us * 1.0e-6 * 24.0 * 299792458.0/tobs/tobs * sqrt((acc_tol*acc_tol)-1.0);
	    unsigned int naccels = (unsigned int)((float)(acc_end-acc_start))/alt_a;
	    acc_list.reserve(naccels+3);
	    if (acc_end>0 && acc_start<0)
		acc_list.push_back(0.0); 
	    float acc = acc_start;
	    while (acc<acc_end){
		acc_list.push_back(acc);
		acc+=alt_a;
	    }
	    acc_list.push_back(acc_end);
	    return;
	}
    } //pipeline
} //peasoup
