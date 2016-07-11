#ifndef FFASTER_DISTILLER_CUH_
#define FFASTER_DISTILLER_CUH_

#include "ffaster.h"
#include "base.cuh"

namespace FFAster
{

    void find_peaks(std::vector<ffa_output_t>& input, 
		    std::vector<ffa_output_t>& output,
		    float thresh=10, 
		    float lthresh=9, 
		    float dthresh=0.2);
    
    void discard_harmonics(std::vector<ffa_output_t>& cands,
			   std::vector<ffa_output_t>& out_cands,
			   float tol = 0.001, 
			   int max_harm = 32);

} //namespace FFAster

#include "detail/distiller.inl"

#endif // FFASTER_DISTILLER_CUH_
