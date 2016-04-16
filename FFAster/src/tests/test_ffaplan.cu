#include "ffaster.h"
#include "ffaplan.cuh"
#include "ffa.cuh"
#include "detrend.cuh"
#include "test_utils.cuh"
#include "io/sigproc.cuh"
#include "io/file.cuh"
#include "io/stream.cuh"
#include "misc/system.cuh"
#include "data_types/timefrequency.cuh"
#include "transforms/dedisperser.cuh"
#include "pipelines/dmtrialqueue.cuh"

using namespace FFAster;

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



int main()
{
    peasoup::io::IOStream* stream = new peasoup::io::FileStream("tmp.fil");
    stream->prepare();
    typedef typename peasoup::type::TimeFrequencyBits<peasoup::HOST> data_type;
    data_type data(0);
    peasoup::io::sigproc::SigprocReader< data_type > reader(data,stream);
    reader.read();
    
    peasoup::type::DispersionTime<peasoup::HOST,uint8_t> dmtrials;
    peasoup::transform::Dedisperser dedisp(data,dmtrials,1);
    printf(data.metadata.display().c_str());
    dedisp.gen_dmlist(180.0,200.0,4000.0,1.80);
    dedisp.prepare();
    dedisp.execute();
    
    peasoup::pipeline::DMTrialQueue<decltype(dmtrials)> queue(dmtrials);
    peasoup::type::TimeSeries<peasoup::HOST,float> host_input;
	
    FFAster::FFAsterPlan<> plan(dmtrials.get_nsamps(), data.metadata.tsamp, 0.001, 0.002, 0.001, 16);
    
    size_t output_bytes = plan.get_required_output_bytes();
    size_t tmp_bytes = plan.get_required_tmp_bytes();
    
    char* tmp_memory;
    Utils::device_malloc<char>(&tmp_memory,tmp_bytes);
    
    ffa_output_t* output;
    Utils::device_malloc<char>((char**)&output,output_bytes);
    plan.set_tmp_storage_buffer((void*) tmp_memory,tmp_bytes);


    ffa_params_t dummy_plan;
    dummy_plan.downsampled_size = dmtrials.get_nsamps();
    Detrender<> detrender;
    detrender.baseline_estimator->power_of_5 = 5; //tweak...
    size_t required_bytes_detrend = detrender.get_required_tmp_bytes(dummy_plan);
    void *tmp_storage_detrend = NULL;
    Utils::device_malloc<char>((char**)&tmp_storage_detrend,required_bytes_detrend);
    detrender.set_tmp_storage_buffer(tmp_storage_detrend,required_bytes_detrend);

    while (queue.pop(host_input)){
	
	peasoup::type::TimeSeries<peasoup::HOST,float> host_input_dereddened = host_input;
	float *in_ = &(host_input.data[0]);
	float *out = &(host_input_dereddened.data[0]);
	running_mean(in_,out,host_input.data.size(),150000);
	peasoup::type::TimeSeries<peasoup::DEVICE,float> device_input = host_input_dereddened;

	Utils::dump_host_buffer<float>(in_,host_input.data.size(),"red.bin");
	Utils::dump_host_buffer<float>(out,host_input.data.size(),"dered.bin");
	
	printf("Procesing trial DM %f\n",device_input.metadata.dm);
	std::stringstream stream;
	stream << "periodogram_" << device_input.metadata.dm << ".bin";
	float* in = thrust::raw_pointer_cast(device_input.data.data());
	//detrender.execute(in,in,dummy_plan);
	plan.execute(in,output);
	Utils::dump_device_buffer<char>((char*)output,output_bytes,stream.str().c_str());
    }
    Utils::device_free(tmp_storage_detrend);
    return 0;

}
