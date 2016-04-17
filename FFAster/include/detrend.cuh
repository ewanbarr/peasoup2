
#ifndef FFASTER_DETREND_CUH_
#define FFASTER_DETREND_CUH_
#define LOG5 1.6094379124341003f

#include "ffaster.h"
#include "base.cuh"
#include "../cub/cub/cub.cuh"

namespace FFAster
{
  namespace Kernels
  {
    
    //Credit to 'DRBlaise'. See http://stackoverflow.com/a/2117018
    __host__ __device__ __forceinline__
    float median_of_5(float a, 
		      float b,
		      float c,
		      float d,
		      float e)
    {
      return b < a ? d < c ? b < d ? a < e ? a < d ? e < d ? e : d
	: c < a ? c : a
	: e < d ? a < d ? a : d
	: c < e ? c : e
	: c < e ? b < c ? a < c ? a : c
	: e < b ? e : b
	: b < e ? a < e ? a : e
	: c < b ? c : b
	: b < c ? a < e ? a < c ? e < c ? e : c
	: d < a ? d : a
	: e < c ? a < c ? a : c
	: d < e ? d : e
	: d < e ? b < d ? a < d ? a : d
	: e < b ? e : b
	: b < e ? a < e ? a : e
	: d < b ? d : b
	: d < c ? a < d ? b < e ? b < d ? e < d ? e : d
	: c < b ? c : b
	: e < d ? b < d ? b : d
	: c < e ? c : e
	: c < e ? a < c ? b < c ? b : c
	: e < a ? e : a
	: a < e ? b < e ? b : e
	: c < a ? c : a
	: a < c ? b < e ? b < c ? e < c ? e : c
	: d < b ? d : b
	: e < c ? b < c ? b : c
	: d < e ? d : e
	: d < e ? a < d ? b < d ? b : d
	: e < a ? e : a
	: a < e ? b < e ? b : e
	: d < a ? d : a;
    }
    
    __global__
    void median_of_5_reduce_k(float* input,
			      float* output,
			      int nreturn,
			      int power_of_5,
			      size_t tot_size);
    
    __global__
    void remove_interpolated_baseline_k(float* input,
					float* output,
					float* medians,
					size_t step,
					size_t size,
					size_t med_size);
    
    __global__ 
    void block_reduce_k(float *input, 
			float *output,
			size_t size,
			bool square=false);
    
        __global__
	void block_reduce_k_2(float *input,
			      float *output,
			      size_t size,
			      bool square=false);
    
    __global__
    void multiply_by_value_k(float *input,
			     float* output,
			     size_t size,
			     float value);

  }; /* namespace Kernels */
  
  /* ABCs */

  template <class TransformType>
  class BaselineEstimator: public TransformType 
  {
  public:
    BaselineEstimator(){}
    virtual void set_smoothing_length(size_t size)=0;
    virtual int get_baseline_step(){return 1;}
    virtual void execute(float *input,
			 float *output,
			 ffa_params_t& plan)=0;
  };

  template <class TransformType>
  class BaselineSubtractor: public TransformType
  {
  protected:
    float* baseline;
    size_t step_size;
    size_t baseline_size;

  public:
    BaselineSubtractor()
      :baseline(NULL),
       step_size(0),
       baseline_size(0)
    {}
    
    virtual void set_baseline(float* baseline_,
			      size_t step_size_,
			      size_t baseline_size_)
    {
      baseline = baseline_;
      step_size = step_size_;
      baseline_size = baseline_size_;
    }

    virtual void execute(float *input,
                         float *output,
                         ffa_params_t& plan)=0;
  };
  
  template <class TransformType>
  class Normaliser: public TransformType
  {
  private:
    virtual float calculate_normalisation(float* input,
                                          size_t size)=0;
    
  public:
    virtual void execute(float* input,
			 float* output,
			 ffa_params_t& plan)=0;
  };

  /* Derived classes */

  /* Derived baseline estimators */
  
  template <class TransformType = Base::DeviceTransform>
  class MedianOfFiveBaselineEstimator: public BaselineEstimator<TransformType>
  {
  public:
    int power_of_5;
    
  public:
    MedianOfFiveBaselineEstimator(int power_of_5_=3)
      :power_of_5(power_of_5_){}
    
    //Rounds up to nearest power of 5
    void set_smoothing_length(size_t size);
    int get_baseline_step();
    size_t get_required_tmp_bytes(ffa_params_t& plan);
    size_t get_required_output_bytes(ffa_params_t& plan);
    void execute(float *input,
		 float* output,
		 ffa_params_t& plan);
  };

  /* Derived normalisers */
  template <class TransformType = Base::DeviceTransform>
  class StdDevNormaliser: public Normaliser<TransformType>
  {
  private:
    //float calculate_normalisation(float* input,
    //				  size_t size);
  public:   
    float calculate_normalisation(float* input,
                                  size_t size);
    size_t get_required_tmp_bytes(ffa_params_t& plan);
    void execute(float* input,
		 float* output,
		 ffa_params_t& plan);
  };
  
  template <class TransformType = Base::DeviceTransform>
  class MadsNormaliser: public Normaliser<TransformType>
  {
  private:
    float calculate_normalisation(float* input,
				  size_t size);
  };
  
  /* Derived baseline subtractors */
  
  template <class TransformType = Base::DeviceTransform>
  class LinearInterpBaselineSubtractor: public BaselineSubtractor<TransformType>
  {
  public:
    void execute(float* input, 
		 float* output,
		 ffa_params_t& plan);
    
    size_t get_required_output_bytes(ffa_params_t& plan);
    
  };
  
  /* helper functions */

  void multiply_by_value(float *input,
			 float* output,
			 size_t size,
			 float value,
			 cudaStream_t stream);
  
  /* Master detrender class 
   * This class wraps and contains all the required elements 
   * for detrending time series used in the FFA */


  
  /* this should be an composed class */
  template <class BaselineSubtractorType = LinearInterpBaselineSubtractor<Base::DeviceTransform>,
	    class BaselineEstimatorType = MedianOfFiveBaselineEstimator<Base::DeviceTransform>,
	    class NormaliserType = StdDevNormaliser<Base::DeviceTransform>,
	    class TransformType = Base::DeviceTransform>
  class Detrender: public TransformType
  {
  public:
    BaselineSubtractorType* baseline_subtractor;
    BaselineEstimatorType* baseline_estimator;
    NormaliserType* normaliser;
    float *baseline;
    size_t baseline_size;
    
    void allocate_internal_memory(ffa_params_t& plan)
    {
      if (this->tmp_storage_bytes == 0)
	throw std::runtime_error
	  ("Detrender::allocate_internal_memory internal memory required.");
      
      char* ptr = (char*) this->tmp_storage;
      size_t bytes;
      
      bytes = baseline_estimator->get_required_tmp_bytes(plan);
      baseline_estimator->set_tmp_storage_buffer((void*)ptr,bytes);
      ptr+=bytes;
      
      bytes = baseline_subtractor->get_required_tmp_bytes(plan);
      baseline_subtractor->set_tmp_storage_buffer((void*)ptr,bytes);
      ptr+=bytes;
	
      bytes = normaliser->get_required_tmp_bytes(plan);
      normaliser->set_tmp_storage_buffer((void*)ptr,bytes);
      ptr+=bytes;
	
      baseline = (float*) ptr;
      ptr += baseline_estimator->get_required_output_bytes(plan);
      
      if (std::distance((char*)this->tmp_storage,ptr) > this->tmp_storage_bytes)
	throw std::runtime_error
	  ("Detrender::allocate_internal_memory insufficuent internal memory allocated.");
    }

  public:
    Detrender()
    {
      baseline_subtractor = new BaselineSubtractorType;
      baseline_estimator = new BaselineEstimatorType;
      normaliser = new NormaliserType;
    }
    
    ~Detrender()
    {
      delete baseline_subtractor;
      delete baseline_estimator;
      delete normaliser;
    }
    
    void set_stream(cudaStream_t stream)
    {
      TransformType::set_stream(stream);
      baseline_estimator->set_stream(stream);
      baseline_subtractor->set_stream(stream);
      normaliser->set_stream(stream);
    }
    
    size_t get_required_tmp_bytes(ffa_params_t& plan)
    {
      size_t nbytes = 0;
      nbytes += baseline_estimator->get_required_tmp_bytes(plan);
      nbytes += baseline_estimator->get_required_output_bytes(plan);
      nbytes += baseline_subtractor->get_required_tmp_bytes(plan);
      nbytes += normaliser->get_required_tmp_bytes(plan);
      return nbytes;
    }

    size_t get_required_output_bytes(ffa_params_t& plan)
    {
      return plan.downsampled_size*sizeof(float);
    }
    
    void execute(float* input,
		 float* output,
		 ffa_params_t& plan)
    {
      allocate_internal_memory(plan);
      size_t baseline_size = (baseline_estimator->get_required_output_bytes(plan))/sizeof(float);
      int step_size = baseline_estimator->get_baseline_step();
      baseline_estimator->execute(input,baseline,plan);
      baseline_subtractor->set_baseline(baseline,step_size,baseline_size);
      baseline_subtractor->execute(input,output,plan);
      normaliser->execute(output,output,plan);
    }
  };

  
}; /* namespace FFAster */

#endif
