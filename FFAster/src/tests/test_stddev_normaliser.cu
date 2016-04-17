#include "test_normaliser.cuh"

using namespace FFAster;

class CPUStdDevNormaliser: public FFAster::Normaliser
{
public:
  float calculate_normalisation(float* input,
				size_t size)
  {
    int ii;
    float mean = 0;
    for (ii=0; ii<size; ii++)
      mean+=input[ii];
    mean/=size;
    
    float std = 0;
    for(ii=0; ii<size; ii++)
      std += (input[ii]-mean) * (input[ii]-mean);
    std = sqrt(std/size);
    return std;
  }
  
  void execute(float* input,
	       ffa_output_t* output,
	       ffa_params_t& plan)
  {
    execute(input,output,plan.downsampled_size);
  }

  void execute(float* input,
	       float* output,
	       size_t size)
  {
    float std = calculate_normalisation(input,size);
    for (int ii=0; ii<size; ii++)
      output[ii] = input[ii]/std;
  }
};

bool test_std_dev_normaliser(int size, float mean, float std)
{
  printf("Testing TestStdDevNormaliser with (%d, %.5f, %.5f)",size,mean,std);
  TestUtils::NormalNumberGenerator genrand(mean,std);
  TestUtils::PulsePattern_f pattern(xdim,ydim,(int)(xdim*0.01),xdim/2,10,0.2,&genrand);
  TestUtils::TestFFAOutputAnalyser<CPUMatchedFilterAnalyser,FFAster::MatchedFilterAnalyser> test_case(xdim,ydim,0.25);
  test_case.set_test_pattern(&pattern);
  bool passed = test_case.test(0.01);
  if (!passed)
    {
      test_case.dump_buffers("TestMatchedFilterAnalyser");
      throw std::runtime_error("Test failed");
    }
  return passed;
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
