#ifndef PEASOUP_ARGS_HPP
#define PEASOUP_ARGS_HPP

#include <vector>
#include <utility>

namespace peasoup {
    namespace pipeline {

	struct Options {
	    std::string format;
	    std::string search_type;
	    std::string infilename;
	    std::string outdir;
	    std::string killfilename;
	    std::string zapfilename;
	    unsigned ngpus;
	    unsigned nthreads;
	    unsigned nfft;
	    float dm_start;
	    float dm_end;
	    float dm_tol;
	    float dm_pulse_width;
	    float acc_start;
	    float acc_end;
	    float acc_tol;
	    float acc_pulse_width;
	    unsigned nharm;
	    float minsigma;
	    float freq_tol;
	    bool verbose;
	    bool progress_bar;
	};
	
	struct AccelSearchArgs
	{
	    std::vector<float> acc_list; // acceleration list to search
	    std::vector<std::pair<float,float> > birdies; // list of RFI frequencies to exclude
            float minsigma; // minimum sigma for periodicity search
            int nharm; // number of harmonics to sum
	    unsigned nfft;//size to pad to.
	};
    }
}

#endif
