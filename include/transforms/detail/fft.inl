#include <stdexcept>
#include <thrust/functional.h>
#include "transforms/fft.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace transform {
	
	template <System system> void RealToComplexFFT<system>::prepare(){
	    size_t size = input.data.size();
	    output.data.resize(size/2 + 1);
	    output.metadata.binwidth = 1.0/(input.metadata.tsamp*size);
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    _prepare();
        }

	template <> void RealToComplexFFT<HOST>::_prepare(){
	    plan = fftwf_plan_dft_r2c_1d(input.data.size(), &(input.data[0]), 
					 (fftwf_complex*) &(output.data[0]), 
					 FFTW_ESTIMATE);    
	    if (plan == NULL)
		throw std::runtime_error("FFTW returned NULL plan.");
	}
	
	template <> void RealToComplexFFT<DEVICE>::_prepare(){
            cufftResult error = cufftPlan1d(&plan, input.data.size(), CUFFT_R2C, 1);
	    utils::check_cufft_error(error);
        }

	template <> void RealToComplexFFT<DEVICE>::execute(){
	    cufftReal* in = thrust::raw_pointer_cast(input.data.data());
	    cufftComplex* out = (cufftComplex*) thrust::raw_pointer_cast(output.data.data());
	    cufftResult error = cufftExecR2C(plan, in, out);
	    utils::check_cufft_error(error);
	    if (normalise) _normalise();
        }

	template <> void RealToComplexFFT<HOST>::execute(){
	    FFTDerivedBase<HOST>::execute();
	    if (normalise) _normalise();
        }
	
	
	template <System system> void RealToComplexFFT<system>::_normalise(){
	    float factor = sqrtf(2.0/input.data.size());
	    auto& out = output.data;
	    thrust::transform(policy_traits.policy, out.begin(), 
			      out.end(), out.begin(), 
			      functor::multiply_by_constant< thrust::complex<float> >(factor));
	}
	
	
	
	template <System system> void ComplexToRealFFT<system>::prepare(){
	    size_t size = input.data.size();
	    size_t new_size = 2*(size - 1);
	    output.data.resize(new_size);
	    output.metadata.tsamp = 1.0/(new_size * input.metadata.binwidth);
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    _prepare();
        }
	
	template <> void ComplexToRealFFT<HOST>::_prepare(){
            plan = fftwf_plan_dft_c2r_1d(output.data.size(), (fftwf_complex*) &(input.data[0]),
                                         &(output.data[0]), FFTW_ESTIMATE);
	    if (plan == NULL)
		throw std::runtime_error("FFTW returned NULL plan.");
        }
	
        template <> void ComplexToRealFFT<DEVICE>::_prepare(){
            cufftResult error = cufftPlan1d(&plan, output.data.size(), CUFFT_C2R, 1);
	    utils::check_cufft_error(error);
        }
	
	template <> void ComplexToRealFFT<HOST>::execute(){
            FFTDerivedBase<HOST>::execute();
	    if (normalise) _normalise();
        }
	
	template <> void ComplexToRealFFT<DEVICE>::execute(){
            cufftComplex* in = (cufftComplex*) thrust::raw_pointer_cast(input.data.data());
	    cufftReal* out = thrust::raw_pointer_cast(output.data.data());
            cufftResult error = cufftExecC2R(plan, in, out);
	    utils::check_cufft_error(error);
	    if (normalise) _normalise();
        }
    
	
	template <System system> void ComplexToRealFFT<system>::_normalise(){
	    using namespace thrust::placeholders;
	    float factor = 1.0/sqrtf(output.data.size()*2);
	    auto& out = output.data;
	    thrust::transform(policy_traits.policy, out.begin(),
			      out.end(), out.begin(),
			      functor::multiply_by_constant<float>(factor));
	}
    } // namespace transform
} // namespace peasoup

