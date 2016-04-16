#ifndef FFASTER_TEST_CUH_
#define FFASTER_TEST_CUH_

#include "ffaster.h"
#include "base.cuh"

namespace FFAster 
{
  namespace TestUtils
  {
    template <class T>
    struct ArrayComparitor
    {
      float tolerance;
      ArrayComparitor(float tolerance=0):tolerance(tolerance){}
      bool operator()(T a, T b){ return abs(a-b)<=tolerance; }
      void print(T a, T b){ printf("|  %.5f  |  %.5f  |\n",(float)a,(float)b); }
    };
    
    template <>
    struct ArrayComparitor<ffa_output_t>
    {
      float tolerance;
      ArrayComparitor(float tolerance=0):tolerance(tolerance){}
      
      bool operator()(ffa_output_t a, ffa_output_t b)
      {
	bool snr_pass = (abs(a.snr-b.snr)<=tolerance);
	bool width_pass = a.width == b.width;
	return (snr_pass && width_pass);
      }
      
      void print(ffa_output_t a, ffa_output_t b){
	printf("S/N |  %.5f  |  %.5f  |        Width |  %d  |  %d  |\n",a.snr,b.snr,a.width,b.width);
      } 
    };
    
    template <class T>
    bool compare_arrays(T* a, 
			T* b, 
			size_t size,
			ArrayComparitor<T>& comparitor,
			bool verbose = false,
			int nfails_to_display = 10)
			
    {
      bool passed = true;
      for (size_t ii=0; ii<size; ii++)
	{
	  if (!comparitor(a[ii],b[ii]))
	    {
	      passed = false;
	      printf("FAILED at idx(%d)  ",ii);
	      comparitor.print(a[ii],b[ii]);
	      nfails_to_display--;
	      if (nfails_to_display == 0)
		break;
	    }
	  else if (verbose)
	    {
	      printf("PASSED at idx(%d)  ",ii);
	      comparitor.print(a[ii],b[ii]);
	    }
	}
      if (passed)
	printf("TEST PASSED SUCCESSFULLY\n");
      else 
	printf("TEST FAILED\n");
      return passed;
    }

    struct NormalNumberGenerator
    {
    private:
      bool have_spare;
      float spare;
      float mean;
      float variance;
      double rand1,rand2;

    public:
      NormalNumberGenerator(float mean=0.0, float variance=1.0)
	:have_spare(false),
	 mean(mean),
	 variance(variance)
      {
	srand(time(NULL));
      }
      
      float get_rand()
      {
	if (have_spare)
	  {
	    have_spare = false;
	    return spare;
	  }
	else
	  {
	    rand1 = rand() / ((double) RAND_MAX);
	    if(rand1 < 1e-100) 
	      rand1 = 1e-100;
	    rand1 = -2 * log(rand1);
	    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;
	    spare = (float) (sqrt(variance * rand1) * cos(rand2) + mean);
	    have_spare = true;
	    return (float) (sqrt(variance * rand1) * sin(rand2) + mean);
	  }
      }
    };
    
    struct TestPattern_f
    {
    public:
      size_t xdim;
      size_t ydim;
      TestPattern_f(size_t xdim, size_t ydim)
	:xdim(xdim),
	 ydim(ydim){}
      
      virtual void operator()(float* ar){
	for (int y=0; y<ydim; y++)
	  for (int x=0; x<xdim; x++)
	    ar[y*xdim+x] = y*xdim+x;
      }
    };

    struct PureNoise_f: public TestPattern_f {
      NormalNumberGenerator generator;
      
      PureNoise_f(size_t xdim, size_t ydim, float mean, float varience)
	:TestPattern_f(xdim,ydim),
	 generator(mean,varience){}
      
      void operator()(float* ar)
      {
	size_t n = xdim*ydim;
	for(int x=0; x<n; x++)
	  ar[x] = generator.get_rand();
      }
    };

    struct TestVector_f: public TestPattern_f {
    public:
      size_t nbytes;
      std::ifstream infile;
      std::string fname;

      TestVector_f(std::string fname)
	:TestPattern_f(0,1),
	 fname(fname)
      {
	infile.open(fname.c_str(),std::ifstream::in | std::ifstream::binary);
	Utils::check_file_error(infile, fname);
	infile.seekg(0, std::ios::end);
	nbytes = infile.tellg();
	infile.close();
	this->xdim = nbytes/sizeof(float);
      }
      
      void operator()(float* ar)
      {
	infile.open(fname.c_str(),std::ifstream::in | std::ifstream::binary);
	Utils::check_file_error(infile, fname);
	infile.read((char*)ar,nbytes);
	infile.close();
      }
    };
    
    struct ChequerBoardPattern_f: public TestPattern_f {
    public:
      ChequerBoardPattern_f(size_t xdim, size_t ydim)
	:TestPattern_f(xdim,ydim){}
      
      void operator()(float* ar)
      {
	for (int y=0; y<ydim; y++)
          for (int x=0; x<xdim; x++)
	    ar[y*xdim+x] = (float)(y%2==x%2); 
      }
    };

    struct PulsePattern_f: public TestPattern_f {
    private:
      size_t pulse_width;
      size_t pulse_phase;
      float pulse_power;
      float slope;
      NormalNumberGenerator *generator;
      
    public:
      PulsePattern_f(size_t xdim,
		     size_t ydim,
		     size_t pulse_width,
		     size_t pulse_phase,
		     float pulse_power,
		     float slope,
		     NormalNumberGenerator* generator=NULL)
	:TestPattern_f(xdim,ydim),
	 pulse_width(pulse_width),
	 pulse_phase(pulse_phase),
	 pulse_power(pulse_power),
	 slope(slope),
	 generator(generator){}
      
      void operator()(float* ar)
      {
	int leading_edge;
	
	for (int y=0; y<ydim; y++)
	  {
	    if (generator!=NULL)
	      {
		for(int x=0; x<xdim; x++)
		  ar[y*xdim+x] = generator->get_rand();
	      }
	    else
	      {
		for(int x=0; x<xdim; x++)
		  ar[y*xdim+x] = 0.0;
	      }
	    
	    leading_edge = ((int)(pulse_phase + slope*y))%((int)xdim);
	    
	    for (int ii=0; ii<pulse_width; ii++)
	      ar[y*xdim + (leading_edge+ii)%((int)xdim)] += pulse_power;
	    
	  }
      }
    };

    template <class InType, class OutType>
    class BaseTestCase
    {
    protected:
      bool pattern_set;
      
    public:
      InType* host_input;
      size_t input_size;

      BaseTestCase(size_t input_size)
        :input_size(input_size),
	 pattern_set(false)
      {
	Utils::host_malloc<InType>(&host_input,input_size);
      }

      ~BaseTestCase()
      {
	Utils::host_free(host_input);
      }
      
      void set_test_pattern(TestPattern_f* functor)
      {
        pattern_set = true;
	functor->operator()(host_input);
      }
      
      virtual void set_test_pattern(BaseTestCase<InType,OutType>* other)
      {
        pattern_set = true;
	Utils::h2hcpy<InType>(host_input,other->host_input,input_size);
      }

    };

    template <typename InType, typename OutType>
    class HostTestCase: public BaseTestCase<InType,OutType>
    {
    public:
      HostTestCase(size_t input_size)
        :BaseTestCase<InType,OutType>(input_size){}
      
      template <class Transform>
      OutType* execute(Transform* transform, ffa_params_t& plan)
      {
        size_t out_bytes = transform->get_required_output_bytes(plan);
        size_t tmp_bytes = transform->get_required_tmp_bytes(plan);
        void* tmp;
        OutType *out;
	Utils::host_malloc<char>((char**)&tmp, tmp_bytes);
	Utils::host_malloc<char>((char**)&out, out_bytes);
        transform->set_tmp_storage_buffer(tmp,tmp_bytes);
	transform->execute(this->host_input,out,plan);
	return out;
      }
    };
    

    template <typename InType, typename OutType>
    class DeviceTestCase:public BaseTestCase<InType,OutType>
    {
    private:
      InType* device_input;
      
    public:
      DeviceTestCase(size_t input_size)
        :BaseTestCase<InType,OutType>(input_size)
      {
	Utils::device_malloc<InType>(&device_input,input_size);
      }

      ~DeviceTestCase()
      {
	Utils::device_free(device_input);
      }
      
      template <class Transform>
      OutType* execute(Transform* transform, ffa_params_t& plan)
      {
	size_t out_bytes = transform->get_required_output_bytes(plan);
	size_t output_size = out_bytes/sizeof(OutType);
	size_t tmp_bytes = transform->get_required_tmp_bytes(plan);
	void* tmp;
	OutType *d_out, *h_out;
	Utils::device_malloc<char>((char**)&tmp, tmp_bytes);
	Utils::device_malloc<char>((char**)&d_out, out_bytes);
	Utils::host_malloc<char>((char**)&h_out, out_bytes);
	transform->set_tmp_storage_buffer(tmp,tmp_bytes);
	Utils::h2dcpy<InType>(device_input,this->host_input,this->input_size);
        transform->execute(device_input,d_out,plan);
	Utils::d2hcpy<OutType>(h_out,d_out,output_size);
	Utils::device_free(tmp);
	Utils::device_free(d_out);
	return h_out;
      }

    };
      
        
    class TestCase
    {
    public:
      float* d_in;
      float* d_out;
      float* h_in;
      float* h_out;
      float* h_d_out;
      size_t insize;
      size_t outsize;
      
      TestCase(size_t insize, size_t outsize)
	:insize(insize),
	 outsize(outsize)
      {
	Utils::device_malloc<float>(&d_in,insize);
	Utils::host_malloc<float>(&h_in,insize);
	Utils::device_malloc<float>(&d_out,outsize);
	Utils::host_malloc<float>(&h_out,outsize);
	Utils::host_malloc<float>(&h_d_out,outsize);
      }
      
      ~TestCase()
      {
	Utils::device_free(d_in);
	Utils::device_free(d_out);
	Utils::host_free(h_in);
	Utils::host_free(h_out);
	Utils::host_free(h_d_out);
      }
      
      void populate(TestPattern_f* functor)
      {
	functor->operator()(h_in);
	Utils::h2dcpy<float>(d_in,h_in,insize);
      }

      void copy_back_results()
      {
	Utils::d2hcpy<float>(h_d_out,d_out,outsize);
      }

    };
  };
};

#endif
