include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_fft_based_pipelines_src
    test_preprocessor.cu
    test_accelsearcher.cu
    test_tf_fft_pipeline.cu
    gtest_fft_based_pipelines.cu
    )

cuda_add_executable(gtest_fft_based_pipelines ${gtest_fft_based_pipelines_src} )
target_link_libraries(gtest_fft_based_pipelines chi2lib)
target_link_libraries(gtest_fft_based_pipelines ${GTEST_LIBRARIES})
target_link_libraries(gtest_fft_based_pipelines ${FFTW_LIBRARIES})
target_link_libraries(gtest_fft_based_pipelines ${CUDA_cufft_LIBRARY})
target_link_libraries(gtest_fft_based_pipelines ${DEDISP_LIBRARIES})
add_test(gtest_fft_based_pipelines gtest_fft_based_pipelines)
