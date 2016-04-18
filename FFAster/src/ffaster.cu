#include "ffaster.h"
#include <string>
#include <iostream>
#include <ctime>
#include <tclap/CmdLine.h>
#include "utils/logging.hpp"
#include "ffaplan.cuh"
#include "ffa.cuh"
#include "detrend.cuh"
#include "io/sigproc.cuh"
#include "io/file.cuh"
#include "io/stream.cuh"
#include "misc/system.cuh"
#include "data_types/timefrequency.cuh"
#include "transforms/dedisperser.cuh"
#include "pipelines/dmtrialqueue.cuh"
#include "pipelines/cmdline.cuh"
#include "pipelines/fft_based/preprocessor.cuh"

namespace FFAster {
    namespace cmdline {

	std::string get_utc_str()
        {
            char buf[128];
	    std::time_t t = std::time(NULL);
	    std::strftime(buf, 128, "./%Y-%m-%d-%H:%M_ffaster/", std::gmtime(&t));
            return std::string(buf);
        }
	
	struct FFAOptions
	{
	    float min_period = 0.8; //minimum period to search in seconds
	    float max_period = 20.0; //maximum period to search in seconds
	    float min_duty_cycle = 0.001; //minimum duty cycle to look for 
	    int nstreams = 16; //number of streams to use (this may have no effect) 
	};

	struct Options 
	{
	    std::string format;
	    std::string infilename;
	    std::string outdir;
	    std::string killfilename;
	    std::string zapfilename;
	    peasoup::pipeline::DedispersionArgs dedispersion;
	    FFAOptions ffa;
	    bool verbose;
	    bool progress_bar;
	};


	
        bool read_cmdline_options(Options& args, int argc, char **argv)
        {
            try {
		TCLAP::CmdLine cmd("FFAster - a GPU FFA-based pulsar search pipeline", ' ', "1.0");


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

		TCLAP::ValueArg<std::string> arg_killfilename("k", "killfile",
                                                              "Channel mask file (only used for filterbank data)",
                                                              false, "", "string",cmd);

		TCLAP::ValueArg<std::string> arg_zapfilename("z", "zapfile",
                                                             "Birdie list file",
                                                             false, "", "string",cmd);

		TCLAP::SwitchArg arg_verbose("v", "verbose", "verbose mode", cmd);

                /*------------------------ FFAOptions ------------------------*/
		
		TCLAP::ValueArg<float> arg_min_period("", "pmin",
                                                    "The minimum period to search for",
                                                    false, args.ffa.min_period, "float",cmd);
		
		TCLAP::ValueArg<float> arg_max_period("", "pmax",
						      "The maximum period to search for",
						      false, args.ffa.max_period, "float",cmd);
		
		TCLAP::ValueArg<float> arg_min_duty_cycle("", "dcmin",
						      "The minimum duty cycle to search for",
						      false, args.ffa.min_duty_cycle, "float",cmd);
		
		
		TCLAP::ValueArg<int> arg_nstreams("t", "nstreams",
                                                  "The number of streams to use per GPU",
                                                  false, args.ffa.nstreams, "int", cmd);


                /*------------------------ DedispersionArgs ------------------------*/
		TCLAP::ValueArg<float> arg_dm_start("", "dm_start",
                                                    "First DM to dedisperse to",
                                                    false, args.dedispersion.dm_start, 
                                                    "float", cmd);

		TCLAP::ValueArg<float> arg_dm_end("", "dm_end",
                                                  "Last DM to dedisperse to",
                                                  false, args.dedispersion.dm_end,
                                                  "float", cmd);

		TCLAP::ValueArg<float> arg_dm_tol("", "dm_tol",
                                                  "DM smearing tolerance (1.11=10%)",
                                                  false, args.dedispersion.dm_tol,
                                                  "float",cmd);

		TCLAP::ValueArg<float> arg_dm_pulse_width("", "dm_pulse_width",
                                                          "Minimum pulse width for which dm_tol is valid",
                                                          false, args.dedispersion.dm_pulse_width,
                                                          "float (us)",cmd);

                cmd.parse(argc, argv);

                //Options
                args.infilename        = arg_infilename.getValue();
                args.outdir            = arg_outdir.getValue();
                args.format            = arg_format.getValue();
                args.killfilename      = arg_killfilename.getValue();
                args.zapfilename       = arg_zapfilename.getValue();
                args.verbose           = arg_verbose.getValue();

                //FFAOptions
                args.ffa.max_period          = arg_max_period.getValue();
                args.ffa.min_period          = arg_min_period.getValue();
		args.ffa.min_duty_cycle      = arg_min_duty_cycle.getValue();
		args.ffa.nstreams            = arg_nstreams.getValue();

                //DedispersionArgs
                args.dedispersion.dm_start          = arg_dm_start.getValue();
                args.dedispersion.dm_end            = arg_dm_end.getValue();
                args.dedispersion.dm_tol            = arg_dm_tol.getValue();
                args.dedispersion.dm_pulse_width    = arg_dm_pulse_width.getValue();

            }catch (TCLAP::ArgException &e) {
                return false;
            }
            return true;
        }
    } // cmdline

    void running_mean(float* in, float* out, size_t size, unsigned window){
	
	unsigned ii,jj;
	double sum;
	size_t count,offset;
	
	for (ii=0;ii<size;ii++)
	    out[ii] = in[ii];
	
	if (window%2 == 0)
	    window +=1;
	
	//Leading edge
	sum = in[0];
	count = 1;
	out[0] = 0.0;
	
	for (ii=1;ii<window/2+1;ii++){
	    sum += in[2*ii-1];
	    sum += in[2*ii];
	    count += 2;
	    out[ii] -= sum/count;
	}
	
	//Middle section
	for (ii=0;ii<size-window;ii++){
	    sum -= in[ii];
	    sum += in[ii+window];
	    out[ii+window/2+1] -= sum/count;
	}
	
	//Trailing edge
	for (ii=size-window/2;ii<size-1;ii++){
	    offset = size-ii-1;
	    sum -= in[size-(2*offset)-3];
	    sum -= in[size-(2*offset)-2];
	    count -= 2;
	    out[ii] -= sum/count;
	}
	out[size-1] = 0.0;
	return;
    }

}//namespace FFaster

int main(int argc, char **argv)
{
    //Read command line
    FFAster::cmdline::Options opts;
    FFAster::cmdline::read_cmdline_options(opts,argc,argv);
    
    //Read input file (only sigproc supported for now)
    peasoup::io::IOStream* stream = new peasoup::io::FileStream(opts.infilename);
    stream->prepare();
    typedef typename peasoup::type::TimeFrequencyBits<peasoup::HOST> data_type;
    data_type data(0);
    peasoup::io::sigproc::SigprocReader< data_type > reader(data,stream);
    reader.read();
    
    // To be update to use  the peasoup logging system
    printf("Read data from file with parameters:\n");
    printf(data.metadata.display().c_str());

    //Dedisperse input
    peasoup::type::DispersionTime<peasoup::HOST,uint8_t> dmtrials;
    peasoup::transform::Dedisperser dedisp(data,dmtrials,1);
    dedisp.gen_dmlist(opts.dedispersion.dm_start,opts.dedispersion.dm_end,
		      opts.dedispersion.dm_pulse_width,opts.dedispersion.dm_tol);
    printf("Dedispersing...");
    dedisp.prepare();
    dedisp.execute();
    printf("done\n");
    
    //Build DM trials queue for processing
    peasoup::pipeline::DMTrialQueue<decltype(dmtrials)> queue(dmtrials);
    
    //Build host vector to read from queue
    peasoup::type::TimeSeries<peasoup::HOST,float> host_input;
    
    //Build FFA plan
    FFAster::FFAsterPlan<> plan(dmtrials.get_nsamps(), data.metadata.tsamp, opts.ffa.min_period,
				opts.ffa.max_period, opts.ffa.min_duty_cycle, opts.ffa.nstreams);
    
    //Allocate all require memory
    size_t output_bytes = plan.get_required_output_bytes();
    size_t tmp_bytes = plan.get_required_tmp_bytes();
    char* tmp_memory;
    FFAster::Utils::device_malloc<char>(&tmp_memory,tmp_bytes);
    FFAster::ffa_output_t* output;
    FFAster::Utils::device_malloc<char>((char**)&output,output_bytes);
    plan.set_tmp_storage_buffer((void*) tmp_memory,tmp_bytes);
    
    //peasoup::type::TimeSeries<peasoup::DEVICE,float> device_red;
    //device_red.data.resize(dmtrials.get_nsamps());
    //device_red.metadata.tsamp = data.metadata.tsamp;

    peasoup::type::TimeSeries<peasoup::DEVICE,float> device_input;
    //peasoup::pipeline::AccelSearchArgs aa_args;
    //aa_args.nfft = dmtrials.get_nsamps();    

    //peasoup::pipeline::Preprocessor<peasoup::DEVICE> preprocessor(device_red,device_input,aa_args);
    //preprocessor.prepare();
    //While there are DM trials left to process
    //pop a DM trial off the queue
    //Currently this is to host memory, due to the
    //mean filter below
    while (queue.pop(device_input)){
    //while (queue.pop(device_red)){

	//Do a running mean to try and alleviate the effects of red noise/
	//peasoup::type::TimeSeries<peasoup::HOST,float> host_input_dereddened = host_input;
	//float *in_ = &(host_input.data[0]);
	//float *out = &(host_input_dereddened.data[0]);
	//int window = opts.ffa.max_period/(2*host_input.metadata.tsamp);
	//FFAster::running_mean(in_,out,host_input.data.size(),window);
	
	

	//Copy dereddened vector to the device for further processing
	//peasoup::type::TimeSeries<peasoup::DEVICE,float> device_input = host_input_dereddened;

	//preprocessor.run();


	//float* red = thrust::raw_pointer_cast(device_red.data.data());
	//FFAster::Utils::dump_device_buffer<float>(red,device_red.data.size(),"red.bin");

	printf("Procesing trial DM %f\n",device_input.metadata.dm);
	std::stringstream stream;
	stream << "periodogram_" << device_input.metadata.dm << ".bin";
	float* in = thrust::raw_pointer_cast(device_input.data.data());

	//Dump input
	//FFAster::Utils::dump_device_buffer<float>(in,device_input.data.size(),"dered.bin");
	
	//Execute FFA
	plan.execute(in,output);

	//Write out periodogram
	FFAster::Utils::dump_device_buffer<char>((char*)output,output_bytes,stream.str().c_str());
    }
    FFAster::Utils::device_free(tmp_memory);
    FFAster::Utils::device_free(output);
    return 0;
}
