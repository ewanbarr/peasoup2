include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_transforms_src
#    test_normaliser.cu
#    test_baselinefinder.cu
#    test_resampler.cu
#    test_harmonicsum.cu
#    test_spectrumformer.cu
#    test_fft.cu
     test_dedisperser.cu
    gtest_transforms.cu
    )

cuda_add_executable(gtest_transforms ${gtest_transforms_src} )
target_link_libraries(gtest_transforms ${GTEST_LIBRARIES})
target_link_libraries(gtest_transforms ${FFTW_LIBRARIES})
target_link_libraries(gtest_transforms ${CUDA_cufft_LIBRARY})
target_link_libraries(gtest_transforms ${DEDISP_LIBRARIES})
add_test(gtest_transforms gtest_transforms)
