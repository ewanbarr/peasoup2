#ifndef FFASTER_H_
#define FFASTER_H_

#include <stdexcept>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include "utils.cuh"
#include "cuda.h"

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif

#ifndef WARP_SIZE 
#define WARP_SIZE 32
#endif

#ifndef LOG2_WARP_SIZE
#define LOG2_WARP_SIZE 5
#endif

#ifndef SQRT2
#define SQRT2 1.414213562f
#endif

#ifndef TWO_PI
#define TWO_PI 6.2831853071795864769252866
#endif

namespace FFAster {

  struct ffa_params_t
  {
    unsigned int downsampling;  //<- dowsampling factor (from original resolution)
    size_t downsampled_size;    //<- size of downsampled data
    double downsampled_tsamp;   //<- sampling time of downsampled data
    float* baseline;
    size_t baseline_size;
    size_t baseline_step;
    unsigned int period_samps;  //<- period in samples
    double period;              //<- period in seconds
    double pstep_samps;         //<- period step in fractions of a sample
    double pstep;               //<- period step in seconds
    unsigned int nturns;        //<- number of valid rotations within the data
    unsigned int nturns_pow2;   //<- next power of two number of rotations
    unsigned int nlayers;       //<- number of butterfly layers in FFA
    unsigned int padded_size;   //<- total size of array with padding
  };

  struct ffa_output_t
  {
    float snr;
    int width;
    float period;
  };
};

#endif
