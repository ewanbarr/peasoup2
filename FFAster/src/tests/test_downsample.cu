/*
 * Downsampler_test.cpp
 *
 *  Created on: 20/10/2014
 *      Author: ebarr
 */

#include "downsample.cuh"
#include "test_utils.cuh"
#include "utils.cuh"

using namespace FFAster;

void cpu_downsample(float* ptr, size_t N, float* output, size_t factor)
{
  size_t ii = 0;
  while (ii<N)
    {
      if ((ii+1)*factor > N)
	break;
      float val = 0.0;
      for (size_t jj=0; jj<factor; jj++)
	val += ptr[ii*factor+jj];
      output[ii] = val;
      ii+=1;
    }
}


int main()
{
  
  //size_t N = 10000000;
  //TestUtils::TestCase test_case(N,N);
  //TestUtils::TestPattern_f generic_pattern(N,1);
  //test_case.populate(&generic_pattern);
  
  TestUtils::TestVector_f test_vector("test_vectors/test_vector_8704000.bin"); 
  size_t N = test_vector.xdim;
  
  TestUtils::TestCase test_case(N,N);
  test_case.populate(&test_vector);
  
  CachedDownsampler dummy_downsampler(NULL,N,32,true);
  for (int ii=11;ii<100;ii++)
    dummy_downsampler.downsample(dummy_downsampler.closest_factor(ii));
  size_t nbytes = dummy_downsampler.get_required_bytes();
  CachedDownsampler downsampler(test_case.d_in,N,32);
  Allocators::SlabAllocator *allocator = new FFAster::Allocators::SlabAllocator(nbytes);
  downsampler.set_allocator(allocator);
  
  for (int ii=11;ii<100;ii++)
    {
      unsigned int factor = downsampler.closest_factor(ii);
      printf("Testing CachedDownsampler with factor %d\n",factor);
      FFAster::CachedDownsampler* result = downsampler.downsample(factor);

      //test_case.copy_back_results();
      Utils::d2hcpy<float>(test_case.h_d_out,result->data,result->size);
      cpu_downsample(test_case.h_in,N,test_case.h_out,factor);
      
      TestUtils::ArrayComparitor<float> comparitor(0.01);
      
      TestUtils::compare_arrays<float>(test_case.h_out,test_case.h_d_out,result->size,comparitor);
    }
  return 0;
}
