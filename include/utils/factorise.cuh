
#ifndef FFASTER_FACTORISER_CUH_
#define FFASTER_FACTORISER_CUH_

#include "ffaster.h"

namespace FFAster
{
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
}; /* namespace FFAster */

#endif
