#ifndef PEASOUP_CHARACTERISTICS_H
#define PEASOUP_CHARACTERISTICS_H

/* Pilfered from Presto (Thanks Scott!) */

namespace peasoup {
    namespace cand_utils {
	double equivalent_gaussian_sigma(double logp);
	/* Return the approximate significance in Gaussian sigmas */
	/* corresponding to a natural log probability logp        */
	
	double chi2_logp(double chi2, int dof);
	/* Return the natural log probability corresponding to a chi^2 value */
	/* of chi2 given dof degrees of freedom. */
	
	double chi2_sigma(double chi2, int dof);
	/* Return the approximate significance in Gaussian sigmas        */
	/* sigmas of a chi^2 value of chi2 given dof degrees of freedom. */
	
	double candidate_sigma(double power, int numsum, double numtrials);
	/* Return the approximate significance in Gaussian       */
	/* sigmas of a candidate of numsum summed powers,        */
	/* taking into account the number of independent trials. */
	
	double power_for_sigma(double sigma, int numsum, double numtrials);
	/* Return the approximate summed power level required */
	/* to get a Gaussian significance of 'sigma', taking  */
	/* into account the number of independent trials.     */
	
	double chisqr(double *data, int numdata, double avg, double var);
	/* Calculates the chi-square of the 'data' which has average */
	/* 'avg', and variance 'var'.                                */
	
	void switch_f_and_p(double in, double ind, double indd,
			    double *out, double *outd, double *outdd);
	/* Convert p, p-dot, and p-dotdot into f, f-dot, */
	/* and f-dotdot or vise-versa.                   */
    } //namespace cand_utils
} //namespace peasoup

#endif
