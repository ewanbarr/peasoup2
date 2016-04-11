#include "transforms/distillers.cuh"

namespace peasoup {
    namespace transform {
	
	inline void DistillerBase::distill(dets_type& cands, dets_type& out_cands)
	{
	    LOG(logging::get_logger("transform.distiller"),logging::DEBUG,
                "Distilling ",cands.size(),"candidates");
	    
	    int ii,idx,start,count;
	    size = cands.size();
	    unique.resize(size,true);
	    std::sort(cands.begin(),cands.end(),
		      [](const type::Detection& a, const type::Detection& b)->bool
		      {  
			  return (a.sigma>b.sigma);  
		      });
	    start = count = 0;
	    while (true) {
		idx = -1;
		for(ii=start;ii<size;ii++){
		    if (unique[ii]){
			start = ii+1;
			idx = ii;
			break;
		    }
		}
		if (idx==-1)
		    break;
		else{
		    count++;
		    condition(cands,idx);
		}
	    }
	    for (ii=0;ii<size;ii++){
		if (unique[ii])
		    out_cands.push_back(cands[ii]);
	    }
	    LOG(logging::get_logger("transform.distiller"),logging::DEBUG,
                "Remaining candidates: ",out_cands.size());
	}
	
	inline HarmonicDistiller::HarmonicDistiller(float tol,float max_harm)
	    :tolerance(tol),max_harm(max_harm){}
	
	inline void HarmonicDistiller::condition(dets_type& cands, int idx)
	{
	    int ii,jj,kk;
	    double ratio,freq;
	    double upper_tol = 1+tolerance;
	    double lower_tol = 1-tolerance;
	    double fundi_freq = cands[idx].freq;
	    int max_frac = pow(2,this->max_harm);
	    for (ii=idx+1;ii<size;ii++){
		freq = cands[ii].freq;
		for (jj=1;jj<=max_frac;jj++){
		    for (kk=1;kk<=max_frac;kk++){
			ratio = kk*freq/(jj*fundi_freq);
			if (ratio>(lower_tol)&&ratio<(upper_tol)){
			    cands[idx].associated.push_back(cands[ii]);
			    unique[ii]=false;
			    goto outer; //<-- legitimate use of goto, omg!
			}
		    }
		}
	    outer:;
	    }
	}
	
	inline AccelerationDistiller::AccelerationDistiller(float tobs, float tol)
	    :tobs(tobs),tolerance(tol),tobs_over_c(tobs/SPEED_OF_LIGHT){}

	inline double AccelerationDistiller::correct_for_acceleration(double freq, double delta_acc)
	{
	    return freq+delta_acc*freq*tobs_over_c;
	}
	
	inline void AccelerationDistiller::condition(dets_type& cands, int idx)
	{
	    int ii;
	    double fundi_freq = cands[idx].freq;
	    double fundi_acc = cands[idx].acc;
	    double acc_freq;
	    double delta_acc;
	    double edge = fundi_freq*tolerance;
	    for (ii=idx+1;ii<size;ii++){
		delta_acc = fundi_acc-cands[ii].acc;
		acc_freq = correct_for_acceleration(fundi_freq,delta_acc);
		if (acc_freq>fundi_freq){
		    if (cands[ii].freq>fundi_freq-edge && cands[ii].freq<acc_freq+edge){
			cands[idx].associated.push_back(cands[ii]);
			unique[ii]=false;
		    }
		} else {
		    if (cands[ii].freq<fundi_freq+edge && cands[ii].freq>acc_freq-edge){
			cands[idx].associated.push_back(cands[ii]);
			unique[ii]=false;
		    }
		}
	    }
	}
	
	inline DMDistiller::DMDistiller(float tol)
	    :tolerance(tol){}
	
	inline void DMDistiller::condition(dets_type& cands,int idx)
	{
	    int ii;
	    double fundi_freq = cands[idx].freq;
	    double upper_tol = 1+tolerance;
	    double lower_tol = 1-tolerance;
	    for (ii=idx+1;ii<size;ii++){
		ratio = cands[ii].freq/fundi_freq;
		if (ratio>(lower_tol)&&ratio<(upper_tol)){
		    cands[idx].associated.push_back(cands[ii]);
		    unique[ii]=false;
		}
	    }
	}
    } // transforms
} // peasoup
