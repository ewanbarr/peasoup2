
#include "detrend.cuh"
#include "downsample.cuh"
#include "test.cuh"

#define MAX_TEST_LEN 10000000
#define MAX_POWER 10

using namespace FFAster;

/*
 * Median of 5 tests
 */

//assumes mean is zero
float cpu_std_dev(float* input, size_t n)
{
  double ssum=0;
  for (int ii=0;ii<n;ii++)
    {
      ssum += (input[ii]*input[ii]);
    }
  return (float) sqrt(ssum/n);
}

void cpu_multiply_by_value(float *input, float value, size_t n)
{
  for (int ii=0;ii<n;ii++)
    input[ii]*=value;
}


void cpu_remove_baseline(float *input, float* output, 
			 float *medians, size_t step,
			 size_t size, size_t med_size)
{						
  float x0, x1, y0, y1;
  int start_idx, end_idx;
  float partial;
  for (int ii=0; ii<med_size-1; ii++)
    {
      if (ii==0)
	{
	  start_idx = 0;
	  end_idx = step + step/2;
	} 
      else if (ii==med_size-2)
	{
	  start_idx = ii*step + step/2;
	  end_idx = size;
	}
      else
	{
	  start_idx = ii*step + step/2;
	  end_idx = (ii+1)*step + step/2;
	}

      x0 = ii*step + step/2.0;
      x1 = (ii+1)*step + step/2.0;
      y0 = medians[ii];
      y1 = medians[ii+1];
      partial = (y1-y0) / (x1-x0);
      
      for (int jj=start_idx; jj<end_idx; jj++)
	{
	  output[jj] = input[jj] - (y0 + partial * (jj-x0));
	}
    }
}


//return num_valid
size_t cpu_median_of_5(float *input, float* output,
		       size_t size, int power_of_5)
{
  float *ptr;
  float median;
  float * tmp_buffer;
  Utils::host_malloc<float>(&tmp_buffer,size);
  Utils::h2hcpy<float>(tmp_buffer,input,size);
  for (int ii=0; ii<power_of_5; ii++)
    {
      for (int jj=0; jj<size/5; jj++)
	{
	  ptr = tmp_buffer + jj*5;
	  median = Kernels::median_of_5(ptr[0],ptr[1],ptr[2],ptr[3],ptr[4]);
	  tmp_buffer[jj] = median;
	}
      size/=5;
    }
  Utils::h2hcpy<float>(output,tmp_buffer,size);
  return size;
}

/*void test_median_of_5(size_t n, int pow_of_5)
{

  printf("Testing median of 5 with (%d, %d)...\n",n,pow_of_5);
  TestUtils::TestCase test_case(n,n);
  TestUtils::TestPattern_f generic_pattern(n,1);
  test_case.populate(&generic_pattern);
  cpu_median_of_5(test_case.h_in,test_case.h_out,n,pow_of_5);
  size_t output_bytes, tmp_storage_bytes;
  MedianOfFiveBaselineEstimator gpu_med_5(pow_of_5);
  output_bytes = gpu_med_5.get_required_output_bytes(n);
  tmp_storage_bytes = gpu_med_5.get_required_tmp_bytes(n);
  fflush(stdout);  
  void *tmp_storage = NULL;
  if (tmp_storage_bytes > 0)
    {
      Utils::device_malloc<char>((char**)&tmp_storage,tmp_storage_bytes);
      gpu_med_5.set_tmp_storage_buffer(tmp_storage,tmp_storage_bytes);
    }
  gpu_med_5.estimate_baseline(test_case.d_in,test_case.d_out,n);
  test_case.copy_back_results();
  
  size_t outlen = output_bytes/sizeof(float);
  bool passed = TestUtils::compare_arrays<float>(test_case.h_d_out,
						 test_case.h_out,
						 outlen,
						 0.01);
  if (tmp_storage)
    Utils::device_free(tmp_storage);
    }*/

void whiten_test_vector(std::string fname, int factor_, int pow_of_5)
{
  TestUtils::TestVector_f test_vector(fname);
  size_t N = test_vector.xdim;
  TestUtils::TestCase test_case(N,N);
  test_case.populate(&test_vector);
  CachedDownsampler dummy_downsampler(NULL,N,32,true);
  unsigned int factor = dummy_downsampler.closest_factor(factor_);
  dummy_downsampler.downsample(factor);
  size_t nbytes = dummy_downsampler.get_required_bytes();
  CachedDownsampler downsampler(test_case.d_in,N,32);
  Allocators::SlabAllocator *allocator = new FFAster::Allocators::SlabAllocator(nbytes);
  downsampler.set_allocator(allocator);
  FFAster::CachedDownsampler* result = downsampler.downsample(factor);
  size_t n = result->size;
  
  ffa_params_t plan;
  plan.downsampled_size = n;
  

  MedianOfFiveBaselineEstimator estimator(3);
  StdDevNormaliser normaliser;
  LinearInterpBaselineSubtractor subtractor;
  Detrender detrender(&subtractor, &estimator, &normaliser);
  size_t required_bytes = detrender.get_required_tmp_bytes(plan);
  
  void *tmp_storage = NULL;
  Utils::dump_device_buffer<float>(result->data,n,"prewhitened_test_vector.bin");
  Utils::device_malloc<char>((char**)&tmp_storage,required_bytes);
  
  detrender.set_tmp_storage_buffer(tmp_storage,required_bytes);
  detrender.execute(result->data,result->data,plan);
  Utils::dump_device_buffer<float>(result->data,n,"whitened_test_vector.bin");
  Utils::device_free(tmp_storage);
}


void test_std_dev(size_t n)
{
  printf("Testing StdDevCalculator with size = %d\n",n);
  TestUtils::TestCase test_case(n,n);
  TestUtils::PureNoise_f pattern(n,1,234.234,3212.0);
  test_case.populate(&pattern);

  ffa_params_t plan;
  plan.downsampled_size = n;
  
  StdDevNormaliser normaliser;
  size_t nbytes = normaliser.get_required_tmp_bytes(plan);
  
  void* tmp_storage = NULL;
  Utils::device_malloc<char>((char**)&tmp_storage,nbytes);
  normaliser.set_tmp_storage_buffer(tmp_storage,nbytes);
  //Utils::dump_host_buffer<float>(test_case.h_in,n,"prenormalisation_test_vector.bin");
 
  float calculate_normalisation(float* input,
				size_t size);

  float gpu_std = normaliser.calculate_normalisation(test_case.d_in,plan.downsampled_size);
  
  normaliser.execute(test_case.d_in,test_case.d_in,plan);
  Utils::d2hcpy<float>(test_case.h_out,test_case.d_in,n);
  float cpu_std = cpu_std_dev(test_case.h_in,n);
  
 
  
  printf("CPU factor: %f     GPU factor: %f\n",cpu_std,gpu_std);

  cpu_multiply_by_value(test_case.h_in,1/cpu_std,n);
  float new_cpu_std = cpu_std_dev(test_case.h_in,n);
  gpu_std = cpu_std_dev(test_case.h_out,n);
  
  if (abs(new_cpu_std-gpu_std)/cpu_std > 0.001)
    printf("failed with size %d, valued %f --- %f\n",n,new_cpu_std,gpu_std);
  else
    printf("test passed\n");
}

void block_reduce_test(size_t n)
{
  
  printf("Testing StdDevCalculator with size = %d\n",n);
  TestUtils::TestCase test_case(n,n);
  TestUtils::PureNoise_f pattern(n,1,234.234,3212.0);
  test_case.populate(&pattern);
  
  StdDevNormaliser normaliser;

  
  
  Kernels::block_reduce_k<<<1,1024,0,0>>>(test_case.d_in,test_case.d_out,n,false);
  
  Utils::d2hcpy<float>(test_case.h_out,test_case.d_out,1);

  printf("value: %f\n",test_case.h_out[0]);

  Kernels::block_reduce_k_2<<<1,1024,0,0>>>(test_case.d_in,test_case.d_out,n,false);
  
  Utils::d2hcpy<float>(test_case.h_out,test_case.d_out,1);

  printf("value: %f\n",test_case.h_out[0]);
}


int main()
{
  //Test the median of 5 calculation for a 
  //variety of inputs
  /*
  for (size_t n=10; n<MAX_TEST_LEN; n*=9)
    for (int power=1; power<MAX_POWER; power++)
      if (n/(int)pow(5,power) > 2)
	test_baseline_removal(n,power);
  */
  
  //whiten_test_vector("tests/test_vector_8704000.bin",105,3);
  //test_median_of_5(82895,3);
  /*
  for (int jj=123; jj<1234567; jj*=2)
    for (int kk=0; kk<10; kk++)
      test_std_dev(jj);
  */
  for (int ii=0; ii<1024; ii++)
    block_reduce_test(ii);


  return 0;
}


