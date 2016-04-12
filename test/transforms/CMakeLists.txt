include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_transforms_src
    test_normaliser.cu
    test_baselinefinder.cu
    test_resampler.cu
    test_harmonicsum.cu
    test_spectrumformer.cu
    test_fft.cu
    test_dedisperser.cu
    test_zapper.cu
    test_pad.cu    
    gtest_transforms.cu
    )

cuda_add_executable(gtest_transforms ${gtest_transforms_src})
target_link_libraries(gtest_transforms chi2lib)
target_link_libraries(gtest_transforms ${GTEST_LIBRARIES})
target_link_libraries(gtest_transforms ${FFTW_LIBRARIES})
target_link_libraries(gtest_transforms ${CUDA_cufft_LIBRARY})
target_link_libraries(gtest_transforms ${DEDISP_LIBRARIES})
add_test(gtest_transforms gtest_transforms)

#-----------test peakfinder------------#

set(
    gtest_transform_peakfinder_src
    test_peakfinder.cu
    )

cuda_add_executable(gtest_transform_peakfinder ${gtest_transform_peakfinder_src})
target_link_libraries(gtest_transform_peakfinder ${GTEST_LIBRARIES})
target_link_libraries(gtest_transform_peakfinder chi2lib)

#---------test peakfinder/distiller--------#
set(
    gtest_transform_distillers_src
    test_distillers.cu
    )

cuda_add_executable(gtest_transform_distillers ${gtest_transform_distillers_src})
target_link_libraries(gtest_transform_distillers ${GTEST_LIBRARIES})
target_link_libraries(gtest_transform_distillers chi2lib)

#-----------test cached downsampler---------#
set(
    gtest_transform_cachedownsampler_src
    test_cachedownsampler.cu
    )

cuda_add_executable(gtest_transform_cachedownsampler ${gtest_transform_cachedownsampler_src})
target_link_libraries(gtest_transform_cachedownsampler ${GTEST_LIBRARIES})
