
#ifndef FFASTER_FFAPLAN_CUH_
#define FFASTER_FFAPLAN_CUH_

#include "ffaster.h"
#include "ffa.cuh"
#include "factorise.cuh"
#include "downsample.cuh"
#include "detrend.cuh"
#include "snengine.cuh"

namespace FFAster
{
  
  void create_ffa_plan(ffa_params_t& params,
		       size_t size,
		       double tsamp,
		       double period,
		       double min_duty_cycle,
		       int max_factor,
		       Factoriser* factoriser=NULL)
  {
    bool own_factoriser = false;
    if (factoriser==NULL)
      {
	factoriser = new Factoriser;
	own_factoriser = true;
      }

    int ideal_downsampling    = (int) floor(period/tsamp * min_duty_cycle);
    params.downsampling       = factoriser->get_nearest_factor(ideal_downsampling,max_factor);
    params.downsampled_size   = size/params.downsampling;
    params.downsampled_tsamp  = tsamp*params.downsampling;
    params.period_samps       = (unsigned int) floor(period/params.downsampled_tsamp);
    params.period             = params.period_samps * params.downsampled_tsamp;
    params.nturns             = params.downsampled_size/params.period_samps;
    params.nlayers            = (unsigned int) ceil( log2( (double) params.nturns ) );
    params.nturns_pow2        = (unsigned int) pow(2, params.nlayers);
    params.padded_size        = params.nturns_pow2 * params.period_samps;
    params.pstep_samps        = (params.period_samps+1) / ( (double) params.padded_size);
    params.pstep              = params.pstep_samps * params.downsampled_tsamp;
    
    /*
      printf("params.downsampling: %d\n"
      "params.downsampled_size: %d\n"
      "params.downsampled_tsamp: %f\n"
      "params.period_samps: %d\nparams.period: %f\n"
      "params.nturns: %d\n"
      "params.nlayers: %d\n"
      "params.nturns_pow2: %d\n"
      "params.padded_size: %d\n"
      "params.pstep_samps: %f\n"
      "params.pstep: %f\n",
      params.downsampling,params.downsampled_size,params.downsampled_tsamp,
      params.period_samps,params.period,params.nturns,params.nlayers,
      params.nturns_pow2,params.padded_size,params.pstep_samps,params.pstep);
    */
    if (own_factoriser)
      delete factoriser;
  }

  
  template <class DetrenderType = Detrender<>,
	    class FFAType = Radix2FFA<>,
	    class AnalyserType = MatchedFilterAnalyser<>,
	    class TransformType = Base::DeviceTransform>
  class FFAsterExecutionUnit: public TransformType
  {
    
  private:
    float* ffa_output;
    
    void allocate_internal_memory(ffa_params_t& plan)
    {
      if (this->tmp_storage_bytes == 0)
	throw std::runtime_error
	  ("FFAsterExecutionUnit::allocate_internal_memory "
	   "internal memory required.");
      
      char* ptr = (char*) this->tmp_storage;
      size_t bytes;
      bytes = detrender->get_required_tmp_bytes(plan);
      detrender->set_tmp_storage_buffer((void*)ptr, bytes);
      ptr += bytes;
      
      bytes = ffa->get_required_tmp_bytes(plan);
      ffa->set_tmp_storage_buffer((void*)ptr,bytes);
      ptr += bytes;
      
      ffa_output = (float*) ptr;
      ptr += ffa->get_required_output_bytes(plan);
      bytes = analyser->get_required_tmp_bytes(plan);
      analyser->set_tmp_storage_buffer((void*)ptr, bytes);
      ptr += bytes;

      if (std::distance((char*)this->tmp_storage,ptr) > this->tmp_storage_bytes)
	throw std::runtime_error
	  ("FFAsterExecutionUnit::allocate_internal_memory "
	   "insufficuent internal memory allocated.");
    }

  public:
    DetrenderType* detrender;
    FFAType* ffa;
    AnalyserType* analyser;
    
    FFAsterExecutionUnit()
    {
      detrender = new DetrenderType;
      ffa = new FFAType;
      analyser = new AnalyserType;
    }
    
    ~FFAsterExecutionUnit()
    {
      delete detrender;
      delete ffa;
      delete analyser;
    }

    size_t get_required_tmp_bytes(ffa_params_t& plan)
    {
      size_t bytes = 0;
      bytes += detrender->get_required_tmp_bytes(plan);
      bytes += ffa->get_required_tmp_bytes(plan);
      bytes += ffa->get_required_output_bytes(plan);
      bytes += analyser->get_required_tmp_bytes(plan);
      return bytes;
    }
    
    size_t get_required_output_bytes(ffa_params_t& plan)
    {
      return analyser->get_required_output_bytes(plan);
    }
    
    void set_stream(cudaStream_t stream_)
    {
      TransformType::set_stream(stream_);
      detrender->set_stream(stream_);
      ffa->set_stream(stream_);
      analyser->set_stream(stream_);
    }

    void execute(float* input, 
		 ffa_output_t* output,
		 ffa_params_t& plan)
    {
      allocate_internal_memory(plan);
      detrender->execute(input,input,plan);
      ffa->execute(input,ffa_output,plan);
      analyser->execute(ffa_output,output,plan);
    }
  };


  template <class FFAsterExecutionUnitType = FFAsterExecutionUnit<>, 
	    class TransformType = Base::DeviceTransform>
  class FFAsterPlan: public TransformType
  {
  private:
    const size_t size;
    const double tsamp;
    const double min_period;
    const double max_period;
    const double min_duty_cycle;
    std::vector< cudaStream_t > streams;
    std::vector< ffa_params_t > plans;
    std::vector< FFAsterExecutionUnitType* > execution_streams; 
    Allocators::ScratchAllocator* downsampling_allocator;
    
    void prepare_streams(int nstreams)
    {
      execution_streams.resize(nstreams);
      streams.resize(nstreams);
      for (int ii=0;ii<nstreams;ii++)
	{
	  cudaStreamCreate(&(streams[ii]));
	  Utils::check_cuda_error("Error creating stream");
	  execution_streams[ii] = new FFAsterExecutionUnitType();
	  execution_streams[ii]->set_stream(streams[ii]);
	}
    }
    
    void prepare_plans()
    {
      plans.resize(0);
      double period = min_period;
      Factoriser* factoriser = new Factoriser;
      while (period < max_period)
	{
	  ffa_params_t plan;
	  FFAster::create_ffa_plan(plan,size,tsamp,period,min_duty_cycle,32,factoriser);
	  plans.push_back(plan);
	  period = plan.period + plan.pstep*(plan.nturns_pow2);
	}
      delete factoriser;
    }
    
    void allocate_internal_memory()
    {
      
      size_t dsamp_bytes = get_required_downsampling_bytes();
      size_t required_bytes_per_stream = get_required_tmp_bytes_per_stream();
      
      if (this->tmp_storage_bytes == 0)
	throw std::runtime_error
	  ("FFAsterPlan::allocate_internal_memory "
	   "internal memory required.");
      
      char* tmp_ptr = (char*) this->tmp_storage;
      downsampling_allocator = new Allocators::ScratchAllocator((void*)tmp_ptr,dsamp_bytes);
      tmp_ptr += dsamp_bytes;
	for (int ii=0; ii<streams.size(); ii++)
	{
	  execution_streams[ii]->set_tmp_storage_buffer((void*)tmp_ptr,required_bytes_per_stream);
	  tmp_ptr+=required_bytes_per_stream;
	}

      if (std::distance((char*)this->tmp_storage,tmp_ptr) > this->tmp_storage_bytes)
	throw std::runtime_error
	  ("FFAsterPlan::allocate_internal_memory "
	   "insufficuent internal memory allocated.");
    }

    size_t get_required_tmp_bytes_per_stream()
    {
      size_t max_bytes = 0;
      size_t nbytes;
      for (int ii=0; ii<plans.size(); ii++)
        {
          fflush(stdout);
          nbytes = execution_streams[0]->get_required_tmp_bytes(plans[ii]);
          if (nbytes>max_bytes)
            max_bytes = nbytes;
        }
      return max_bytes;
    }
    
    size_t get_required_downsampling_bytes()
    {
      CachedDownsampler dummy_downsampler(NULL,size,32,true);
      for (int ii=0; ii<plans.size(); ii++)
	dummy_downsampler.downsample(plans[ii].downsampling);
      return dummy_downsampler.get_required_bytes();
    }
    
  public:
    FFAsterPlan(const size_t size_,
		const double tsamp_,
		const double min_period_,
		const double max_period_,
		const double min_duty_cycle_,
		const int nstreams)
      :size(size_),
       tsamp(tsamp_),
       min_period(min_period_),
       max_period(max_period_),
       min_duty_cycle(min_duty_cycle_)
    {
      prepare_streams(nstreams);
      prepare_plans();
    }
    
    ~FFAsterPlan(){
      for (int ii=0; ii<streams.size(); ii++)
	{
	  cudaStreamDestroy(streams[ii]);
	  //delete execution_streams[ii];
	}
    }
    
    size_t get_required_tmp_bytes()
    {
      size_t nbytes = get_required_downsampling_bytes();
      nbytes += get_required_tmp_bytes_per_stream() * streams.size();
      return nbytes;
    }

    size_t get_required_output_bytes()
    {
      size_t nbytes = 0;
      
      for (int ii=0; ii<plans.size(); ii++)
	{
	  fflush(stdout);
	  nbytes += execution_streams[0]->get_required_output_bytes(plans[ii]);
	}
      return nbytes;
    }
        
    void execute(float* input, ffa_output_t* output)
    {
      
      Utils::dump_device_buffer<float>(input,size,"input.bin");
      
      allocate_internal_memory();
      CachedDownsampler downsampler(input,size,32);
      downsampler.set_allocator(downsampling_allocator);
      for (int ii=0; ii<plans.size(); ii++)
	downsampler.downsample(plans[ii].downsampling);      
      int nstreams = streams.size();
      size_t offset = 0;
      for (int ii=0; ii<plans.size(); ii++)
	{
	  CachedDownsampler* downsampled = downsampler.downsample(plans[ii].downsampling);
	  execution_streams[ii%nstreams]->execute(downsampled->data,output+offset,plans[ii]);
	  offset += plans[ii].nturns_pow2;
	}
    }
  };
  
}; /* namespace FFAster */

#endif
