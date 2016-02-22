#include "utils/chi2lib.hpp"

extern "C" {
    
    double py_candidate_sigma(double power, int numsum, double numtrials, bool nn)
    {
	return peasoup::cand_utils::candidate_sigma(power,numsum,numtrials,nn);
    }
    
    double py_power_for_sigma(double sigma, int numsum, double numtrials, bool nn)
    {
	return peasoup::cand_utils::power_for_sigma(sigma, numsum,numtrials,nn);
    }
}
