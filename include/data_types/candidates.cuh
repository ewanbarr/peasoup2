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
	    Detection(float freq, float power, int nh,
		      float acc, float dm)
		:freq(freq),power(power),nh(nh),acc(acc),dm(dm){}
	};
	
    } // namespace type
} // namespace peasoup

#endif //PEASOUP_METADATA_H
