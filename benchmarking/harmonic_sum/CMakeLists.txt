set( HARMSUM_BENCH_NVCC_FLAGS "-O3 -Xptxas -dlcm=ca")
set( CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} ${HARMSUM_BENCH_NVCC_FLAGS}" )

set(
    benchmark_harmonic_sum_src
    benchmark_harmonic_sum.cu
    )

cuda_add_executable(benchmark_harmonic_sum ${benchmark_harmonic_sum_src})
target_link_libraries(benchmark_harmonic_sum ${NVTX_LIBRARIES})
