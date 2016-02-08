add_definitions(-DUSE_NVTX)

#-------Harmonic sum-------

set( HARMSUM_BENCH_NVCC_FLAGS "-O3 -Xptxas -dlcm=ca --use_fast_math")
set( CUDA_NVCC_FLAGS  "${CUDA_NVCC_FLAGS} ${HARMSUM_BENCH_NVCC_FLAGS}" )

set(
    benchmark_harmonic_sum_src
    benchmark_harmonic_sum.cu
    )

cuda_add_executable(benchmark_harmonic_sum ${benchmark_harmonic_sum_src})
target_link_libraries(benchmark_harmonic_sum ${NVTX_LIBRARIES})

#--------FFT----------------

set(
    benchmark_fft_src
    benchmark_fft.cu
    )

cuda_add_executable(benchmark_fft ${benchmark_fft_src})
target_link_libraries(benchmark_fft ${CUDA_cufft_LIBRARY})
target_link_libraries(benchmark_fft ${NVTX_LIBRARIES})

#---------resample-----------

set(
    benchmark_resample_src
    benchmark_resample.cu
    )

cuda_add_executable(benchmark_resample ${benchmark_resample_src})
target_link_libraries(benchmark_resample ${NVTX_LIBRARIES})

#--------Spectrum former--------
		  
set(
    benchmark_spectrumform_src
    benchmark_spectrumform.cu
    )

cuda_add_executable(benchmark_spectrumform ${benchmark_spectrumform_src})
target_link_libraries(benchmark_spectrumform ${NVTX_LIBRARIES})


#--------Peak finder-----------


set(
    benchmark_peakfind_src
    benchmark_peakfind.cu
    )

cuda_add_executable(benchmark_peakfind ${benchmark_peakfind_src})
target_link_libraries(benchmark_peakfind ${NVTX_LIBRARIES})
target_link_libraries(benchmark_peakfind chi2lib)


#--------Dedispersion-----------

set(
    benchmark_dedispersion_src
    benchmark_dedispersion.cu
    )

cuda_add_executable(benchmark_dedispersion ${benchmark_dedispersion_src})
target_link_libraries(benchmark_dedispersion ${NVTX_LIBRARIES})
target_link_libraries(benchmark_dedispersion ${DEDISP_LIBRARIES})
