#ifndef PEASOUP_FACTORISER_HPP
#define PEASOUP_FACTORISER_HPP

namespace peasoup{
    namespace utils{
	class Factoriser
	{
	private:
	    std::map< unsigned int, std::vector<unsigned int> > factors;
	    void find_factors(unsigned int factor);
	    
	public:
	    unsigned int first_factor(unsigned int factor);
	    unsigned int get_nearest_factor(unsigned int factor,
					    unsigned int max_factor);
	    std::vector<unsigned int> get_factors(unsigned int factor);
	};
    
	void Factoriser::find_factors(unsigned int factor)
	{
	    int n = factor;
	    int ii = 2;
	    while (ii*ii <= n)
		if (n%ii)
		    ii+=1;
		else {
		    n/=ii;
		    factors[factor].push_back(ii);
		}
	    if (n>1)
		factors[factor].push_back(n);
	    return;
	}

	unsigned int Factoriser::first_factor(unsigned int factor)
	{
	    int n = factor;
	    int ii = 2;
	    while (1)
		if (n%ii == 0)
		    break;
		else
		    ii++;
	    return ii;
	}
	
	unsigned int Factoriser::get_nearest_factor(unsigned int factor,
						    unsigned int max_factor)
	{
	    if (factor==1)
		return 1;
	    if (!factors.count(factor))
		find_factors(factor);
	    if (*std::max_element(factors[factor].begin(),factors[factor].end()) < max_factor)
		return factor;
	    else
		return get_nearest_factor(factor-1,max_factor);
	}
	
	std::vector<unsigned int> Factoriser::get_factors(unsigned int factor)
	{
	    if (!factors.count(factor))
		find_factors(factor);
	    return factors[factor];
	}
    } //namespace utils
}//namespace peasoup

#endif //PEASOUP_FACTORISER_HPP
