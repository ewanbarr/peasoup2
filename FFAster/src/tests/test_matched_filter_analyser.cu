#include "test_utils.cuh"
#include "../snengine.cuh"

using namespace FFAster;

bool test_matched_filter_analyser(int xdim, int ydim)
{
  printf("Testing MatchedFilterAnalyser with (%d,%d)\n",xdim,ydim);
  TestUtils::DeviceTestCase<float,ffa_output_t> d_case(xdim*ydim);
  TestUtils::HostTestCase<float,ffa_output_t> h_case(xdim*ydim);
  TestUtils::NormalNumberGenerator genrand(0.0,1.0);
  TestUtils::PulsePattern_f pattern(xdim,ydim,(int)(xdim*0.01),xdim/2,10,0.2,&genrand);
  h_case.set_test_pattern(&pattern);
  d_case.set_test_pattern(&h_case);

  typedef FFAster::MatchedFilterAnalyser<Base::DeviceTransform> mfa_d;
  typedef FFAster::MatchedFilterAnalyser<Base::HostTransform> mfa_h;
  
  mfa_d d_transform(0.25);
  mfa_h h_transform(0.25);
  
  ffa_params_t plan;
  plan.period_samps = xdim;
  plan.nturns_pow2  = ydim;
  plan.nturns       = ydim;
  
  ffa_output_t *d_out = d_case.execute< mfa_d >(&d_transform,plan);
  ffa_output_t *h_out = h_case.execute< mfa_h >(&h_transform,plan);
  
  TestUtils::ArrayComparitor<ffa_output_t> comparitor(0.01);
  bool passed = TestUtils::compare_arrays<ffa_output_t>(d_out,h_out,ydim,comparitor);
  if (!passed)
    throw std::runtime_error("Test failed");
}

int main()
{
  for (int x=128; x<1500; x*=3.14)
    {
      for (int y=2; y<1024; y*=2)
	test_matched_filter_analyser(x,y);
    }
  return 0;
}
