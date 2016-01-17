#ifndef PEASOUP_TIMSERIES_GENERATOR_CUH
#define PEASOUP_TIMSERIES_GENERATOR_CUH

#include <math.h>

#include <thrust/complex.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>

#include "misc/system.cuh"
#include "misc/constants.h"
#include "data_types/timeseries.cuh"

namespace peasoup {
    namespace generator {
	
	template <typename T>
	void make_noise(type::TimeSeries<HOST,T>& input, float mean=0.0, 
			float std=1.0, float redness=0.0)
	{
	    thrust::minstd_rand rng;
	    thrust::random::normal_distribution<T> dist(mean,std);
	    thrust::generate(thrust::host,input.data.begin(),input.data.end(),
			     [&](){return (T) dist(rng);});
	    
	    for (int ii=1; ii<input.data.size(); ii++){
		input.data[ii] = redness*input.data[ii-1] + 
		    sqrt(1-redness*redness) * input.data[ii];
	    }

	    for (auto& val: input.data){
		val += dist(rng);
		val /= (sqrt(2)*std);
	    }
	}
	
	template <typename T>
	void add_tone(type::TimeSeries<HOST,T>& input, float frequency, 
		       float amplitude=1.0, float phase=0.0, float purity=1.0)
	{
	    float factor = TWOPI * input.metadata.tsamp * frequency;
	    float offset = phase*TWOPI;
	    auto& in = input.data;
	    for (int ii=0;ii<in.size();ii++){
		float val = 1+sinf(ii*factor+offset)/2.0;
		in[ii] += amplitude * powf(val,purity);
	    }
	}
	
	/* 
	   Add a top-hat pulse train to a noisy timeseries.
	   The snr argument assumes that the input time series
	   has mean=0 and std=1. This is the snr of the output
	   pulse in the time domain.
	*/
	template <typename T>
	void add_pulse_train(type::TimeSeries<HOST,T>& input, float frequency,
			      float snr=10.0, float duty_cycle=0.1)
	{
	    float period = 1/frequency;
	    float tsamp = input.metadata.tsamp;
	    float tobs = tsamp*input.data.size();
	    float width = period * duty_cycle;
	    float nrot = tobs/period;
	    float amp = snr/(sqrtf(nrot)*sqrtf(width/tsamp));
	    float phase,rot;
	    auto& in = input.data;
            for (int ii=0;ii<in.size();ii++){
		phase = modf(ii*tsamp/period,&rot);
		if (phase < duty_cycle)
		    in[ii] += amp;
	    }
	}
	
    } //namespace generator
} //namespace peasoup

#endif //PEASOUP_TIMSERIES_GENERATOR_CUH
