#ifndef PEASOUP_ARGS_HPP
#define PEASOUP_ARGS_HPP

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "pipelines/accel_plan.hpp"

namespace peasoup {
    namespace pipeline {

	struct AccelSearchArgs
	{
	    typedef std::pair<float,float> freq_width_type;
	    std::vector<freq_width_type> birdies; // list of RFI frequencies to exclude
	    std::vector<float> user_acc_list;
	    std::shared_ptr<AccelerationPlan> acc_plan;
	    float acc_start       = 0.0;
            float acc_end         = 0.0;
            float acc_tol         = 1.05;
            float acc_pulse_width = 40.0;
	    float chbw            = 0.0;   // channel bandwidth for acc_plan
	    float cfreq           = 1400.0;// centre frequency for acc_plan
            float minsigma        = 5.0;   // minimum sigma for periodicity search
            int nharm             = 4;     // number of harmonics to sum
	    unsigned nfft         = 0;     //size to pad to.
	    bool nn               = true;  //use nearest neighbour spectrum forming.
	    float min_freq        = 0.1;   //minimum frequency to search (def = 10 seconds period)
	    float hdistill_tol    = 0.001; //frequency tolerance for harmonic distiller
	    float adistill_tol    = 0.001; //frequency tolerance for acceleration distiller
	};
	
	struct DedispersionArgs {
	    std::vector<float> dm_list; //dm trial list to search (if empty trials are created)
	    float dm_start       = 0.0;
            float dm_end         = 0.0;
            float dm_tol         = 1.05;
            float dm_pulse_width = 40.0;
	    std::vector<int> chan_mask;
	};

	struct TimeFrequencyFFTPipelineArgs
	{
	    int ngpus = 1;
            int nthreads = 1;
	    DedispersionArgs dedispersion;
	    AccelSearchArgs accelsearch;
	    
	};
	
	struct Options {
	    std::string format;
	    std::string search_type;
	    std::string infilename;
	    std::string outdir;
	    std::string killfilename;
	    std::string zapfilename;
	    TimeFrequencyFFTPipelineArgs tf_fft_args;
            bool verbose;
	    bool progress_bar;
        };

    }
}

#endif
