include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_pipelines_src
    test_dmtrialqueue.cu
    test_worker.cu
    gtest_pipelines.cu
    )

cuda_add_executable(gtest_pipelines ${gtest_pipelines_src} )
target_link_libraries(gtest_pipelines chi2lib)
target_link_libraries(gtest_pipelines ${GTEST_LIBRARIES})
target_link_libraries(gtest_pipelines ${FFTW_LIBRARIES})
target_link_libraries(gtest_pipelines ${CUDA_cufft_LIBRARY})
target_link_libraries(gtest_pipelines ${DEDISP_LIBRARIES})
add_test(gtest_pipelines gtest_pipelines)

add_subdirectory(fft_based)
