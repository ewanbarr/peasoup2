include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_tvgs_src
    test_timeseries_generator.cu
    gtest_tvgs.cu
    )

cuda_add_executable(gtest_tvgs ${gtest_tvgs_src} )
target_link_libraries(gtest_tvgs ${GTEST_LIBRARIES})
add_test(gtest_tvgs gtest_tvgs)
