
#include "ffaster.h"
#include "ffaplan.cuh"
#include "ffa.cuh"
#include "test.cuh"

using namespace FFAster;




int main()
{
  int xdim = 1053;
  int ydim = 1024;
  
  size_t n = xdim * ydim;
  TestUtils::NormalNumberGenerator genrand(0.0,1.0);
  TestUtils::TestCase test_case(n,n);
  TestUtils::PulsePattern_f pattern(xdim,ydim,(int)(xdim*0.01),xdim/2,10,0.2,&genrand);
  test_case.populate(&pattern);

  ffa_params_t ffa;
  ffa.downsampling       = 32;
  ffa.downsampled_size   = n;
  ffa.downsampled_tsamp  = 0.000064*32;;
  ffa.period_samps       = xdim;
  ffa.period             = ffa.period_samps * ffa.downsampled_tsamp;
  ffa.nturns             = ffa.downsampled_size/ffa.period_samps;
  ffa.nlayers            = (unsigned int) ceil( log2( (double) ffa.nturns ) );
  ffa.nturns_pow2        = (unsigned int) pow(2, ffa.nlayers);
  ffa.padded_size        = ffa.nturns_pow2 * ffa.period_samps;
  ffa.pstep_samps        = (ffa.period_samps+1) / ( (double) ffa.padded_size);
  ffa.pstep              = ffa.pstep_samps * ffa.downsampled_tsamp;

  Radix2FFA ffaobj;
  size_t tmp_bytes = ffaobj.get_required_tmp_bytes(ffa);
  size_t out_bytes = ffaobj.get_required_output_bytes(ffa);
  void* ptr;
  float* output;
  Utils::device_malloc<char>((char**)&ptr,tmp_bytes);
  ffaobj.set_tmp_storage_buffer((void*)ptr, tmp_bytes);
  Utils::device_malloc<float>(&output,ffa.padded_size);
  Utils::dump_device_buffer<float>(test_case.d_in,n,"pre_ffa.bin");
  ffaobj.execute(test_case.d_in,output,ffa);
  Utils::dump_device_buffer<float>(test_case.d_out,ffa.padded_size,"post_ffa.bin");
  return 0;
}
