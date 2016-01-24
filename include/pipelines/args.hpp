#ifndef PEASOUP_ARGS_HPP
#define PEASOUP_ARGS_HPP

#include <vector>
#include <utility>

namespace peasoup {
    namespace pipeline {
	struct AccelSearchArgs
	{
	    std::vector<float> acc_list; // acceleration list to search
	    std::vector<std::pair<float,float> > birdies; // list of RFI frequencies to exclude
            float minsigma; // minimum sigma for periodicity search
            int nharm; // number of harmonics to sum
	};
    }
}

#endif
