include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_data_types_src
    test_container.cu
    test_timeseries.cu
    test_timefrequency.cu
    test_dispersiontime.cu
    test_frequencyseries.cu
    test_harmonicseries.cu
    gtest_data_types.cu
    )

cuda_add_executable(gtest_data_types ${gtest_data_types_src} )
target_link_libraries(gtest_data_types ${GTEST_LIBRARIES})
add_test(gtest_data_types gtest_data_types)
