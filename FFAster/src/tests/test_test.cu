
#include "test.cuh"

using namespace FFAster;

void generate_patterns(size_t xdim, size_t ydim)
{
  size_t total_size = xdim * ydim;
  
  TestUtils::TestCase test_case(total_size,total_size);
  TestUtils::NormalNumberGenerator generator(0.0,1.0);
  TestUtils::TestPattern_f generic_pattern(xdim,ydim);
  TestUtils::ChequerBoardPattern_f chequer_pattern(xdim,ydim);
  TestUtils::PulsePattern_f pulse_pattern(xdim,ydim,xdim/100,20,5,-0.2,&generator);
  
  test_case.populate(&generic_pattern);
  Utils::dump_host_buffer<float>(test_case.h_in,total_size,"generic_pattern.bin");

  test_case.populate(&chequer_pattern);
  Utils::dump_host_buffer<float>(test_case.h_in,total_size,"chequer_pattern.bin");
  
  test_case.populate(&pulse_pattern);
  Utils::dump_host_buffer<float>(test_case.h_in,total_size,"pulse_pattern.bin");
}


int main(int argc, char **argv)
{
  size_t xdim,ydim;
  if (argc < 3)
    generate_patterns(1024,2048);
  else
    {
      xdim = atoi(argv[1]);
      ydim = atoi(argv[2]);
      generate_patterns(xdim,ydim);
    }
  return 0;
}
