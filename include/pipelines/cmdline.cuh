#ifndef PEASOUP_CMDLINE_HPP
#define PEASOUP_CMDLINE_HPP

#include <string>
#include <iostream>
#include <ctime>
#include <tclap/CmdLine.h>
#include "pipelines/args.hpp"

namespace peasoup {
    namespace cmdline {

	std::string get_utc_str()
	{
	    char buf[128];
	    std::time_t t = std::time(NULL);
	    std::strftime(buf, 128, "./%Y-%m-%d-%H:%M_peasoup/", std::gmtime(&t));
	    return std::string(buf);
	}
	
	
	bool read_cmdline_options(pipeline::Options& args, int argc, char **argv)
	{
	    try {
		TCLAP::CmdLine cmd("Peasoup2 - a GPU pulsar search pipeline", ' ', "2.0");
	

		/*------------------------ Options ------------------------*/
		TCLAP::ValueArg<std::string> arg_infilename("i", "inputfile",
							    "Name of file to process",
							    true, "", "string", cmd);
		
		TCLAP::ValueArg<std::string> arg_outdir("o", "outdir",
							"Output directory (will be created if it does not exist)",
							false, "", "string",cmd);
		
		TCLAP::ValueArg<std::string> arg_format("f", "format",
							"format for input data (def=sigproc)",
							false, "sigproc", "string", cmd);
		
		TCLAP::ValueArg<std::string> arg_search_type("s", "searchtype",
							     "type of pipeline to use (def=tf_fft)",
							     false, "tf_fft", "string",cmd);
		
		TCLAP::ValueArg<std::string> arg_killfilename("k", "killfile",
							      "Channel mask file (only used for filterbank data)",
							      false, "", "string",cmd);
		
		TCLAP::ValueArg<std::string> arg_zapfilename("z", "zapfile",
							     "Birdie list file",
							     false, "", "string",cmd);
		
		TCLAP::SwitchArg arg_verbose("v", "verbose", "verbose mode", cmd);

		TCLAP::SwitchArg arg_progress_bar("p", "progress_bar", "Enable progress bar for DM search", cmd);

		/*------------------------ TimeFrequencyFFTPipelineArgs ------------------------*/
		TCLAP::ValueArg<int> arg_ngpus("g", "ngpus",
					       "The number of GPUs to use",
					       false, args.tf_fft_args.ngpus, "int", cmd);
		
		TCLAP::ValueArg<int> arg_nthreads("t", "num_threads",
						  "The number of threads to use per GPU",
						  false, args.tf_fft_args.nthreads, "int", cmd);
		
		
		/*------------------------ DedispersionArgs ------------------------*/
		TCLAP::ValueArg<float> arg_dm_start("", "dm_start",
						    "First DM to dedisperse to",
						    false, args.tf_fft_args.dedispersion.dm_start, 
						    "float", cmd);
		
		TCLAP::ValueArg<float> arg_dm_end("", "dm_end",
						  "Last DM to dedisperse to",
						  false, args.tf_fft_args.dedispersion.dm_end,
						  "float", cmd);
		
		TCLAP::ValueArg<float> arg_dm_tol("", "dm_tol",
						  "DM smearing tolerance (1.11=10%)",
						  false, args.tf_fft_args.dedispersion.dm_tol,
						  "float",cmd);
		
		TCLAP::ValueArg<float> arg_dm_pulse_width("", "dm_pulse_width",
							  "Minimum pulse width for which dm_tol is valid",
							  false, args.tf_fft_args.dedispersion.dm_pulse_width,
							  "float (us)",cmd);
		
		/*------------------------ AccelSearchArgs ------------------------*/
		TCLAP::ValueArg<float> arg_acc_start("", "acc_start",
						     "First acceleration to resample to",
						     false, args.tf_fft_args.accelsearch.acc_start,
						     "float", cmd);
		
		TCLAP::ValueArg<float> arg_acc_end("", "acc_end",
						   "Last acceleration to resample to",
						   false, args.tf_fft_args.accelsearch.acc_end, 
						   "float", cmd);

		TCLAP::ValueArg<float> arg_acc_tol("", "acc_tol",
						   "Acceleration smearing tolerance (1.11=10%)",
						   false, args.tf_fft_args.accelsearch.acc_tol,
						   "float",cmd);
		
		TCLAP::ValueArg<float> arg_acc_pulse_width("", "acc_pulse_width",
							   "Minimum pulse width for which acc_tol is valid",
							   false, args.tf_fft_args.accelsearch.acc_pulse_width,
							   "float (us)",cmd);

		TCLAP::ValueArg<int> arg_nharm("n", "nharm",
					       "Number of harmonic sums to perform",
					       false, args.tf_fft_args.accelsearch.nharm, "int", cmd);
		
		
		TCLAP::ValueArg<float> arg_minsigma("m", "minsigma",
						    "The minimum significance threshold for a candidate",
						    false, args.tf_fft_args.accelsearch.minsigma, "float",cmd);

		TCLAP::ValueArg<size_t> arg_nfft("", "nfft",
						 "The length of FFT to perform",
						 false, args.tf_fft_args.accelsearch.nfft, 
						 "size_t",cmd);
		
		TCLAP::ValueArg<float> arg_min_freq("", "min_freq",
						    "The minimum frequency signal to search for",
						    false, args.tf_fft_args.accelsearch.min_freq, "float",cmd);
		
		cmd.parse(argc, argv);
		
		//Options
		args.infilename        = arg_infilename.getValue();
		args.outdir            = arg_outdir.getValue();
		args.search_type       = arg_search_type.getValue();
		args.format            = arg_format.getValue();
		args.killfilename      = arg_killfilename.getValue();
		args.zapfilename       = arg_zapfilename.getValue();
		args.verbose           = arg_verbose.getValue();
		args.progress_bar      = arg_progress_bar.getValue();
		
		//TimeFrequencyFFTPipelineArgs
		args.tf_fft_args.ngpus             = arg_ngpus.getValue();
		args.tf_fft_args.nthreads          = arg_nthreads.getValue();
		
		//DedispersionArgs
		args.tf_fft_args.dedispersion.dm_start          = arg_dm_start.getValue();
		args.tf_fft_args.dedispersion.dm_end            = arg_dm_end.getValue();
		args.tf_fft_args.dedispersion.dm_tol            = arg_dm_tol.getValue();
		args.tf_fft_args.dedispersion.dm_pulse_width    = arg_dm_pulse_width.getValue();
		
		//AccelSearchArgs
		args.tf_fft_args.accelsearch.acc_start         = arg_acc_start.getValue();
		args.tf_fft_args.accelsearch.acc_end           = arg_acc_end.getValue();
		args.tf_fft_args.accelsearch.acc_tol           = arg_acc_tol.getValue();
		args.tf_fft_args.accelsearch.acc_pulse_width   = arg_acc_pulse_width.getValue();
		args.tf_fft_args.accelsearch.nharm             = arg_nharm.getValue();
		args.tf_fft_args.accelsearch.minsigma          = arg_minsigma.getValue();
		args.tf_fft_args.accelsearch.nfft              = arg_nfft.getValue();
		args.tf_fft_args.accelsearch.min_freq          = arg_min_freq.getValue();
		
	    }catch (TCLAP::ArgException &e) {
		std::cerr << "Error: " << e.error() << " for arg " << e.argId()
			  << std::endl;
		return false;
	    }
	    return true;
	}
    } // cmdline
}// peasoup

#endif
