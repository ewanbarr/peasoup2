#ifndef PEASOUP_FFT_CUH
#define PEASOUP_FFT_CUH

#include "thrust/complex.h"
#include "thrust/functional.h"

#include "cufft.h"
#include "thirdparty/fftw3.h"

#include "misc/system.cuh"
#include "data_types/timeseries.cuh"
#include "data_types/frequencyseries.cuh"
#include "transforms/transform_base.cuh"
#include "utils/printer.hpp"

namespace peasoup {
    namespace transform {
	namespace functor {
	    
	    template <typename T>
	    struct multiply_by_constant: thrust::unary_function<T,T>
	    {
		float constant;
		multiply_by_constant(float constant)
		    :constant(constant){}
		inline __host__ __device__
		T operator()(T val) const {return val*constant;}
	    };
		
	} // namespace functor

	
	template <System system>
	class FFTDerivedBase: public Transform<system>
	{  
	};

	template <>
        class FFTDerivedBase<HOST>: public Transform<HOST>
        {
        protected:
            fftwf_plan plan;
            FFTDerivedBase():plan(0){}
            ~FFTDerivedBase(){ if (plan==0) fftwf_destroy_plan(plan); }
        public:
            virtual void execute(){ fftwf_execute(plan); }
        };

        template <>
        class FFTDerivedBase<DEVICE>: public Transform<DEVICE>
        {
        protected:
            cufftHandle plan;
	    SystemPolicy<DEVICE> policy_traits;
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
	    SystemPolicy<system> policy_traits;
	    bool normalise;
	    void _prepare();
	    void _normalise();
	    
	public:
	    RealToComplexFFT(type::TimeSeries<system, float >& input,
			     type::FrequencySeries<system, thrust::complex<float> >& output,
			     bool normalise=true)
		:input(input),output(output),normalise(normalise){}
	    void prepare();
	    void execute();
	};
	
	template <System system>
        class ComplexToRealFFT: public FFTDerivedBase<system>
        {
        private:
	    type::FrequencySeries<system, thrust::complex<float> >& input;
	    type::TimeSeries<system, float >& output;
	    SystemPolicy<system> policy_traits;
	    bool normalise;
	    void _prepare();
	    void _normalise();

        public:
	    ComplexToRealFFT(type::FrequencySeries<system, thrust::complex<float> >& input,
			     type::TimeSeries<system, float >& output,
			     bool normalise=true)
		:input(input),output(output),normalise(normalise){}
            void prepare();
	    void execute();
        };

    } //namespace transform
} //namespace peasoup

#include "transforms/detail/fft.inl"

#endif //PEASOUP_FFT_CUH
