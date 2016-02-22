#ifndef PEASOUP_DISTILLERS_CUH
#define PEASOUP_DISTILLERS_CUH

#include <vector>
#include "data_types/candidates.cuh"
#include "utils/chi2lib.hpp"

namespace peasoup {
    namespace transform {
	typedef std::vector<type::Detection> dets_type;
	
	class DistillerBase 
	{
	protected:
	    std::vector<bool> unique;
	    int size;
	    virtual void condition(dets_type& cands, int idx) = 0;

	public:
	    void distill(dets_type& cands, dets_type& out_cands);
	};
	

	class HarmonicDistiller: public DistillerBase
	{
	private:
	    float tolerance;
	    float max_harm;

	protected:
	    void condition(dets_type& cands, int idx);
	
	public:
	    HarmonicDistiller(float tol, float max_harm);
	};
	
	class AccelerationDistiller: public DistillerBase 
	{
	private:
	    float tobs;
	    double tobs_over_c;
	    float tolerance;
	    
	protected:
	    double correct_for_acceleration(double freq, double delta_acc);	    
	    void condition(dets_type& cands,int idx);
	    
	public:
	    AccelerationDistiller(float tobs, float tolerance);
	};
	
	
	class DMDistiller: public DistillerBase 
	{
	private:
	    float tolerance;
	    double ratio;
	    
	protected:
	    void condition(dets_type& cands,int idx);

	public:
	    DMDistiller(float tol);
	};
	
    } // transform
} // peasoup

#include "transforms/detail/distillers.inl"

#endif //PEASOUP_DISTILLERS_CUH
