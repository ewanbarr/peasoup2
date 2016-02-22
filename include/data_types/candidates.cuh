#ifndef PEASOUP_CANDIDATES_CUH
#define PEASOUP_CANDIDATES_CUH

#include <vector>

namespace peasoup {
    namespace type {
	
	struct Detection
	{
	    float freq;
	    float power;
	    int nh;
	    float acc;
	    float dm;
	    float sigma;
	    std::vector<Detection> associated;
	    Detection(float freq, float power, int nh,
		      float acc, float dm, float sigma=0.0)
		:freq(freq),power(power),nh(nh),
		 acc(acc),dm(dm),sigma(sigma){}
	};


    } // namespace type
} // namespace peasoup

#endif //PEASOUP_METADATA_H
