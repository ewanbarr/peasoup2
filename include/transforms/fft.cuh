#ifndef PEASOUP_FFT_CUH
#define PEASOUP_FFT_CUH

#include "thrust/complex.h"

#include "misc/system.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"

namespace peasoup {
    namespace transform {
	
	class FFTBase
	{
	public:
	    virtual void prepare()=0;
	    virtual void execute()=0;
	};
	
	template <System system>
	class FFTDerivedBase: public FFTBase
	{  
	};

	template <>
        class FFTDerivedBase<HOST>: public FFTBase
        {
        protected:
            fftwf_plan plan;
            FFTDerivedBase():plan(0){}
            ~FFTDerivedBase(){ if (plan==0) fftwf_destroy_plan(plan); }
        public:
            virtual void execute(){ fftwf_execute(plan); }
        };

        template <>
        class FFTDerivedBase<DEVICE>: public FFTBase
        {
        protected:
            cufftHandle plan;
            FFTDerivedBase():plan(0){}
            ~FFTDerivedBase(){ if (plan==0) cufftDestroy(plan); }
	    virtual void execute()=0;
	};

	// only supports float
	template <System system>
	class RealToComplexFFT: public FFTDerivedBase<system>
	{
	private:
	    type::TimeSeries<system, float >& input;
	    type::FrequencySeries<system, thrust::complex<float> >& output;
	    void _prepare();
	    
	public:
	    RealToComplexFFT(type::TimeSeries<system, float >& input,
			     type::FrequencySeries<system, thrust::complex<float> >& output)
		:input(input),output(output){}
	    void prepare();
	    void execute();
	};
	
	template <System system>
        class ComplexToRealFFT: public FFTDerivedBase<system>
        {
        private:
	    type::FrequencySeries<system, thrust::complex<float> >& input;
	    type::TimeSeries<system, float >& output;
	    void _prepare();

        public:
	    ComplexToRealFFT(type::FrequencySeries<system, thrust::complex<float> >& input,
			    type::TimeSeries<system, float >& output)
		:input(input),output(output){}
            void prepare();
	    void execute();
        };

    } //namespace transform
} //namespace peasoup

#include "transforms/detail/fft.inl"

#endif //PEASOUP_FFT_CUH
