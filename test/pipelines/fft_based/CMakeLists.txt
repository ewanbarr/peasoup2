include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

#---------Preprocessor-----------

set(
    gtest_fft_based_preprocessor_src
    test_preprocessor.cu
    )

cuda_add_executable(gtest_fft_based_preprocessor ${gtest_fft_based_preprocessor_src} )
target_link_libraries(gtest_fft_based_preprocessor ${GTEST_LIBRARIES})
target_link_libraries(gtest_fft_based_preprocessor ${FFTW_LIBRARIES})
target_link_libraries(gtest_fft_based_preprocessor ${CUDA_cufft_LIBRARY})
add_test(gtest_fft_based_preprocessor gtest_fft_based_preprocessor)


#-----------Accelsearch---------


set(
    gtest_fft_based_accelsearch_src
    test_accelsearcher.cu
    )

cuda_add_executable(gtest_fft_based_accelsearch ${gtest_fft_based_accelsearch_src} )
target_link_libraries(gtest_fft_based_accelsearch chi2lib)
target_link_libraries(gtest_fft_based_accelsearch ${GTEST_LIBRARIES})
target_link_libraries(gtest_fft_based_accelsearch ${FFTW_LIBRARIES})
target_link_libraries(gtest_fft_based_accelsearch ${CUDA_cufft_LIBRARY})
add_test(gtest_fft_based_accelsearch gtest_fft_based_accelsearch)

#-----------TFPipeline-----------

set(
    gtest_fft_based_tf_pipeline_src
    test_tf_fft_pipeline.cu
    )

cuda_add_executable(gtest_fft_based_tf_pipeline ${gtest_fft_based_tf_pipeline_src} )
target_link_libraries(gtest_fft_based_tf_pipeline chi2lib)
target_link_libraries(gtest_fft_based_tf_pipeline ${GTEST_LIBRARIES})
target_link_libraries(gtest_fft_based_tf_pipeline ${FFTW_LIBRARIES})
target_link_libraries(gtest_fft_based_tf_pipeline ${CUDA_cufft_LIBRARY})
target_link_libraries(gtest_fft_based_tf_pipeline ${DEDISP_LIBRARIES})
add_test(gtest_fft_based_tf_pipeline gtest_fft_based_tf_pipeline)
