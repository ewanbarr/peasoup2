#include "ffaplan.cuh"
#include "test.cuh"

using namespace FFAster;

int main()
{
  int xdim = 1053;
  int ydim = 1021;
  size_t n = xdim * ydim;
  TestUtils::NormalNumberGenerator genrand(4.0,50.0);
  TestUtils::TestCase test_case(n,n);
  TestUtils::PulsePattern_f pattern(xdim,ydim,(int)(xdim*0.01),xdim/2,10,0.2,&genrand);
  test_case.populate(&pattern);
  
  float tsamp = 0.000064;

  ffa_params_t params;
  params.downsampling       = 32;
  params.downsampled_size   = n;
  params.downsampled_tsamp  = tsamp*params.downsampling;
  params.period_samps       = xdim;
  params.period             = params.period_samps * params.downsampled_tsamp;
  params.nturns             = params.downsampled_size/params.period_samps;
  params.nlayers            = (unsigned int) ceil( log2( (double) params.nturns ) );
  params.nturns_pow2        = (unsigned int) pow(2, params.nlayers);
  params.padded_size        = params.nturns_pow2 * params.period_samps;
  params.pstep_samps        = (params.period_samps+1) / ( (double) params.padded_size);
  params.pstep              = params.pstep_samps * params.downsampled_tsamp;
  
  printf("params.downsampling: %d\n"
         "params.downsampled_size: %d\n"
         "params.downsampled_tsamp: %f\n"
         "params.period_samps: %d\n"
         "params.period: %f\n"
         "params.nturns: %d\n"
         "params.nlayers: %d\n"
         "params.nturns_pow2: %d\n"
         "params.padded_size: %d\n"
         "params.pstep_samps: %f\n"
         "params.pstep: %f\n",
         params.downsampling,params.downsampled_size,params.downsampled_tsamp,
         params.period_samps,params.period,params.nturns,params.nlayers,
         params.nturns_pow2,params.padded_size,params.pstep_samps,params.pstep);  
  
  FFAsterExecutionStream* exec_stream = new_ffa_execution_stream
    <MedianOfFiveBaselineEstimator,
    LinearInterpBaselineSubtractor,
    StdDevNormaliser,
    Radix2FFA,
    MatchedFilterAnalyser>(0);
  
  void* tmp_ptr;
  size_t tmp_bytes = exec_stream->get_required_tmp_bytes(params);
  printf("tmp bytes required: %u\n",tmp_bytes);
  Utils::device_malloc<char>((char**)&tmp_ptr,tmp_bytes);
  exec_stream->set_tmp_storage_buffer(tmp_ptr,tmp_bytes);
  
  ffa_output_t* output_ptr;
  size_t output_bytes = exec_stream->get_required_output_bytes(params);
  Utils::device_malloc<char>((char**)&output_ptr,output_bytes);
  
  exec_stream->set_stream(0);
  
  exec_stream->execute(test_case.d_in,output_ptr,params);
  
  Utils::dump_device_buffer<ffa_output_t>(output_ptr,output_bytes/sizeof(ffa_output_t),"ffa_execution_stream_output.bin");
  
}
