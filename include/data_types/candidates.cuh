#ifndef PEASOUP_CANDIDATES_CUH
#define PEASOUP_CANDIDATES_CUH

#include <string>
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

	std::string detections_as_string(std::vector<Detection>& dets)
	{
	    std::stringstream stream;
	    stream << "Freq\tPower\tsigma\tharmonic\tdm\tacc\n";
	    for (auto det: dets)
		stream << det.freq<<"\t"<<det.power<<"\t"
		       <<det.sigma<<"\t"<<det.nh<<"\t"
		       <<det.dm<<"\t"<<det.acc<<"\n";
	    return stream.str();
	}

    } // namespace type
} // namespace peasoup

#endif //PEASOUP_METADATA_H
