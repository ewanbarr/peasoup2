#ifndef PEASOUP_ACCEL_PLAN_HPP
#define PEASOUP_ACCEL_PLAN_HPP

#include <vector>

namespace peasoup {
    namespace pipeline {
	
	class AccelerationPlan 
	{
	public:
	    virtual void get_accelerations(std::vector<float>& acc_list, float dm=0.0f) = 0;
	};
	
	class StaticAccelerationPlan: public AccelerationPlan
	{
	private:
	    std::vector<float> list;
	    
        public:
	    StaticAccelerationPlan(std::vector<float> accs)
		:list(accs){}
            void get_accelerations(std::vector<float>& acc_list, float dm=0.0f);
	};

	class DMDependentAccelerationPlan: public AccelerationPlan
	{
	private:
	    float acc_start;
	    float acc_end;
	    float acc_tol;
	    float acc_pulse_width;
	    size_t nsamps;
	    float tsamp;
	    float cfreq;
	    float cfreq_GHz;
	    float bw;
	    float tsamp_us;
	    float tobs;
	    
	public:
	    DMDependentAccelerationPlan(float acc_start, float acc_end, float acc_tol, 
					float acc_pulse_width, size_t nsamps, float tsamp, 
					float cfreq, float bw);
	    void get_accelerations(std::vector<float>& acc_list, float dm=0.0f);
	};
    } //pipeline
} //peasoup

#include "pipelines/detail/accel_plan.inl"

#endif //PEASOUP_ACCEL_PLAN_HPP
