#idndef PEASOUP_ACCELSEARCHER_CUH
#define PEASOUP_ACCELSEARCHER_CUH

#include <algorithm>

namespace peasoup {
    namespace pipeline {
	namespace worker {
	   
	    using namespace type;
	    using namespace transform;
	    
	    template <System system, typename T>
            class TimeSeriesPreprocessor
            {
	    private:
		typedef thrust::complex<T> complex;
		type::TimeSeries<system,T>& input;
		type::TimeSeries<system,T>& output;
		type::TimeSeries<system,complex> fourier;
		type::FrequencySeries<system,T> spectrum;
		type::FrequencySeries<system,T> baseline;
		transform::Zapper<system,T>* zapper;
		transforms::Normaliser<system,T>* normaliser;
		transforms::SpectrumFormer<system,T>* spectrum_former;
		transforms::BaselineFinder<system,T>* baseline_finder;
		transforms::RealToComplexFFT<system,T> r2cfft;
		transforms::ComplexToRealFFT<system,T>* c2rfft;
		PeasoupArgs args;

	    public:
		TimeSeriesPreprocessor(type::TimeSeries<system,T>& input,
				       type::TimeSeries<system,T>& output,
				       PeasoupArgs args)
		    :input(input),output(output),args(args)
		{
		    float max_accel = *std::max_element(args.acc_list.begin(),args.acc_list.end());
		    if (max_accel<100): max_accel = 100.0;
		    r2cfft = new RealToComplexFFT<system,T>(input,fourier);
		    spectrum_former = new SpectrumFormer<system,T>(fourier,spectrum);
		    baseline_finder = new BaselineFinder<system,T>(spectrum,baseline,max_accel);
		    normaliser = new Normaliser<system,T>(fourier,fourier,baseline);
		    zapper = new Zapper<system,T>(fourier,args.birdies);
		    c2rfft = ComplexToRealFFT<system,T>(fourier,output);
		}

		void prepare()
		{
		    r2cfft.prepare();
		    spectrum_former.prepare();
		    baseline_finder.prepare();
		    normaliser.prepare();
		    zapper.prepare();
		    c2rfft.prepare();
		}

		void run()
		{
		    r2cfft.execute();
		    spectrum_former.form();
		    baseline_finder.find_baseline();
		    normaliser.normalise();
		    zapper.zap();
		    c2rfft.execute();
		}
	    };
		



	    template <System system, typename T>
	    class AccelSearch
	    {
	    private:
		typedef thrust::complex<T> complex;
		type::TimeSeries<system,T> timeseries ;
		type::TimeSeries<system,T> resampled_timeseries;
		type::FrequencySeries<system,complex> amplitudes;
		type::FrequencySeries<system,T> spectrum;
		type::HarmonicSeries<system,T> harmonics;
		transforms::RealToComplexFFT<system,T> r2cfft;
		transforms::RealToComplexFFT<system,T> r2cfft_stage2;
		transforms::ComplexToRealFFT<system,T>* c2rfft;
		transforms::HarmonicSum<system,T>* harmsum;
		transforms::TimeDomainResampler<system,T>* resampler;
		transforms::Zapper<system,T>* zapper;
		transforms::SpectrumFormer<system,T>* spectrum_former;
		transforms::BaselineFinder<system,T>* baseline_finder;
		transforms::Normaliser<system,T>* normaliser;
		transforms::PeakFinder<system,T>* peak_finder;
		
		AccelSearchArgs args;

	    public:
		
		AccelSearch(AccelSearchArgs args)
		    :args(args),
		     r2cfft(timeseries,amplitudes),
		     spectrum_former()
		     baseline_finder(amplitudes,baseline,500.0),
		     zapper(amplitudes,args.birdies),
		     c2rfft(amplitudes,timeseries),
		     resampler(timeseries,resampled_timeseries)
		{
		    
		    

		}

		void prepare(){
		    b
		    

		}
		
	    };

 
	} //worker
    } //pipeline
} //peasoup


#endif // PEASOUP_ACCELSEARCHER_CUH
