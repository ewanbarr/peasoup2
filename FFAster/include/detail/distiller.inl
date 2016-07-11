#include "distiller.cuh"
#include <vector>
using namespace FFAster;

void FFAster::find_peaks(std::vector<ffa_output_t>& input, 
			 std::vector<ffa_output_t>& output,
			 float thresh, float lthresh, float dthresh)
{
    bool above_threshold = false;
    bool peak_flag = false;
    ffa_output_t best = {0,0,0};
    float trough_snr = 0.0;
    
    //sort input by period
    std::sort(input.begin(),input.end(),
	      [](const ffa_output_t& a, const ffa_output_t& b)->bool
	      { return (a.period<b.period); });
    
    for (auto row: input){
	
	//printf("Period: %f   SNR: %f  Peak: %d   Above: %d\n",
	//row.period,row.snr,peak_flag,above_threshold);
	if (above_threshold){
	    
	    if (row.snr < lthresh){
		// the obvious case - if we've fallen below the lower threshold, we're done
		// write out the peak, unless we've already done so
		if (!peak_flag){
		    //printf("Writing peak!\n");
		    output.push_back(best);
		}
		//printf("above_thresh=false\n\n");
		above_threshold = false;
		peak_flag = false;
	    }
	    else if (row.snr > best.snr && !peak_flag){
		//still climbing
		best = row;
	    }
	    else if (row.snr < (1-dthresh)*best.snr && !peak_flag){
		// we have fallen sufficiently far from the highest_snr to classify it as its own peak
		peak_flag = true;
		trough_snr = row.snr;
		//printf("Writing peak!  Period: %f   SNR: %f\n",best.period,best.snr);
		output.push_back(best);
	    }
	    else if (peak_flag && row.snr < trough_snr){
		// we have fallen deeper into the valley since the previous peak
		trough_snr = row.snr;
	    }
	    else if (peak_flag && row.snr > (1+dthresh)*trough_snr && row.snr > thresh){
		// we have climbed sufficiently far out of the valley to call the new ridge its own peak
		peak_flag = false;
		best = row;
	    }
	} else if (!above_threshold){
	    // determine if we have just now crossed the threshold
	    if (row.snr > thresh){
		//printf("above_thresh=true\n\n");
		best = row;
		above_threshold = true;
	    }
	    // if we have not crossed the threshold, nothing to be done
	}
    }
    return;
}
    

void match_harmonics(std::vector<ffa_output_t>& cands, 
		     int idx,
		     std::vector<bool>& unique,
		     float tolerance,
		     float max_frac)
{
    int ii,jj,kk;
    double ratio,period;
    double upper_tol = 1+tolerance;
    double lower_tol = 1-tolerance;
    double fundi_period = cands[idx].period;
    size_t size = cands.size();

    for (ii=idx+1;ii<size;ii++){
	period = cands[ii].period;
	
	for (jj=1;jj<=max_frac;jj++){
	    
	    for (kk=1;kk<=max_frac;kk++){
	
		ratio = kk*period/(jj*fundi_period);
		
		if ( ratio>(lower_tol) && ratio<(upper_tol) ){
		    unique[ii]=false;
		    goto outer; //<-- legitimate use of goto!
		}
	    }
	}
    outer:;
    }
}

void FFAster::discard_harmonics(std::vector<ffa_output_t>& cands,
				std::vector<ffa_output_t>& out_cands,
				float tol, int max_harm)
{
    int ii,idx,start,count;
    size_t size = cands.size();
    
    //set up mask to exclude harmonic matches
    std::vector<bool> unique(cands.size(),true);
    
    //sort the vector bt S/N
    std::sort(cands.begin(),cands.end(),
              [](const ffa_output_t& a, const ffa_output_t& b)->bool
              { return (a.snr>b.snr); });
    
    //while there are still non-matched candidates
    //seek through and look for harmonic matches
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
	    match_harmonics(cands,idx,unique,tol,max_harm);
	}
    }
    for (ii=0;ii<size;ii++){
	if (unique[ii])
	    out_cands.push_back(cands[ii]);
    }
    return;
}
    
    
