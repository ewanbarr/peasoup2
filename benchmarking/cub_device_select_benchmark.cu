#include <cub/cub.cuh>
#include <stdio.h>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cuda.h"
#include "utils/timer.cuh"
#define DSIZE 1<<23

typedef typename cub::ArgIndexInputIterator<int*>::value_type Tuple;

struct GreaterThan
{
    int threshold;
    GreaterThan(int threshold):threshold(threshold){}

    __host__ __device__ __forceinline__
    bool operator()(const Tuple &a) const {
        return (a.value > threshold);
    }
};

int main(){

    int num_items = DSIZE;
    int *d_in;
    Tuple* d_out;
    int *d_num_selected;
    int *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;


    thrust::host_vector<int> h_in(num_items,0);
    for (int n=0;n<num_items;n+=100)
	h_in[n] = 10;
    thrust::device_vector<int> d_in_vec = h_in;
    cudaDeviceSynchronize();
    thrust::device_vector<Tuple> d_out_vec(num_items);
    thrust::device_vector<int> d_num_selected_vec(1);
    
    d_in = thrust::raw_pointer_cast(&d_in_vec[0]);
    d_out = thrust::raw_pointer_cast(&d_out_vec[0]);
    d_num_selected = thrust::raw_pointer_cast(&d_num_selected_vec[0]);
    
    peasoup::utils::Timer clock;
    cub::ArgIndexInputIterator<int *> input_itr(d_in);
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_itr, d_out, d_num_selected, num_items, GreaterThan(2));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    clock.start();
    for (int ii=0;ii<100;ii++)
	cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input_itr, d_out, d_num_selected, num_items, GreaterThan(2));
    clock.stop();
    printf("100 iterations takes %f ms (%f per iteration)\n",clock.elapsed(),clock.elapsed()/100);
    return 0;
}
