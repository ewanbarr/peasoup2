#include <stdexcept>
#include <thrust/functional.h>
#include "transforms/fft.cuh"
#include "utils/utils.cuh"

namespace peasoup {
    namespace transform {
	
	template <System system> void RealToComplexFFT<system>::prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
		"Preparing RealToComplexFFT\n",
		"Normalise FFT: ",normalise,"\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    size_t size = input.data.size();
	    output.data.resize(size/2 + 1);
	    output.metadata.binwidth = 1.0/(input.metadata.tsamp*size);
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    _prepare();
	    
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Prepared RealToComplexFFT\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
	}

	template <> 
	inline void RealToComplexFFT<HOST>::_prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Generating FFTW dft_r2c_1d plan");
	    plan = fftwf_plan_dft_r2c_1d(input.data.size(), &(input.data[0]), 
					 (fftwf_complex*) &(output.data[0]), 
					 FFTW_ESTIMATE);    
	    if (plan == NULL){
		LOG(logging::get_logger("transform.fft"),logging::CRITICAL,"FFTW returned NULL plan.");
		throw std::runtime_error("FFTW returned NULL plan.");
	    }
	}
	
	template <> 
	inline void RealToComplexFFT<DEVICE>::_prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Generating cuFFT 1D R2C plan");
            cufftResult error = cufftPlan1d(&plan, input.data.size(), CUFFT_R2C, 1);
	    if (this->stream!=nullptr)
		cufftSetStream(plan,this->stream);
	    utils::check_cufft_error(error);
        }

	template <> 
	inline void RealToComplexFFT<DEVICE>::execute(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Executing device R2C FFT");
	    cufftReal* in = thrust::raw_pointer_cast(input.data.data());
	    cufftComplex* out = (cufftComplex*) thrust::raw_pointer_cast(output.data.data());
	    cufftResult error = cufftExecR2C(plan, in, out);
	    utils::check_cufft_error(error);
	    if (normalise) _normalise();
        }

	template <> 
	inline void RealToComplexFFT<HOST>::execute(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Executing host R2C FFT");
	    FFTDerivedBase<HOST>::execute();
	    if (normalise) _normalise();
        }
	
	
	template <System system> void RealToComplexFFT<system>::_normalise(){
	    float factor = sqrtf(2.0/input.data.size());
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Normalising R2C FFT output by sqrt(2/N) = ",factor);
	    auto& out = output.data;
	    thrust::transform(policy_traits.policy, out.begin(), 
			      out.end(), out.begin(), 
			      functor::multiply_by_constant< thrust::complex<float> >(factor));
	}
	
	
	
	template <System system> void ComplexToRealFFT<system>::prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Preparing ComplexToRealFFT\n",
                "Normalise FFT: ",normalise,"\n",
                "Input metadata:\n",input.metadata.display(),
                "Input size: ",input.data.size()," samples");
	    size_t size = input.data.size();
	    size_t new_size = 2*(size - 1);
	    output.data.resize(new_size);
	    output.metadata.tsamp = 1.0/(new_size * input.metadata.binwidth);
	    output.metadata.dm = input.metadata.dm;
	    output.metadata.acc = input.metadata.acc;
	    _prepare();
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Prepared ComplexToRealFFT\n",
                "Output metadata:\n",output.metadata.display(),
                "Output size: ",output.data.size()," samples");
        }
	
	template <> 
	inline void ComplexToRealFFT<HOST>::_prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Generating FFTW dft_c2r_1d plan");
            plan = fftwf_plan_dft_c2r_1d(output.data.size(), (fftwf_complex*) &(input.data[0]),
                                         &(output.data[0]), FFTW_ESTIMATE);
	    if (plan == NULL)
		throw std::runtime_error("FFTW returned NULL plan.");
        }
	
        template <> 
	inline void ComplexToRealFFT<DEVICE>::_prepare(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
                "Generating cuFFT 1D C2R plan");
            cufftResult error = cufftPlan1d(&plan, output.data.size(), CUFFT_C2R, 1);
	    if (this->stream!=nullptr)
		cufftSetStream(plan,this->stream);
	    utils::check_cufft_error(error);
        }
	
	template <> 
	inline void ComplexToRealFFT<HOST>::execute(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
		"Executing host C2R FFT");
            FFTDerivedBase<HOST>::execute();
	    if (normalise) _normalise();
        }
	
	template <> 
	inline void ComplexToRealFFT<DEVICE>::execute(){
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
		"Executing device C2R FFT");
            cufftComplex* in = (cufftComplex*) thrust::raw_pointer_cast(input.data.data());
	    cufftReal* out = thrust::raw_pointer_cast(output.data.data());
            cufftResult error = cufftExecC2R(plan, in, out);
	    utils::check_cufft_error(error);
	    if (normalise) _normalise();
        }
	
	
	template <System system> 
	inline void ComplexToRealFFT<system>::_normalise(){
	    float factor = 1.0/sqrtf(output.data.size()*2);
	    LOG(logging::get_logger("transform.fft"),logging::DEBUG,
		"Normalising C2R FFT output by 1/sqrt(2N) = ",factor);
	    auto& out = output.data;
	    thrust::transform(policy_traits.policy, out.begin(),
			      out.end(), out.begin(),
			      functor::multiply_by_constant<float>(factor));
	}
    } // namespace transform
} // namespace peasoup

