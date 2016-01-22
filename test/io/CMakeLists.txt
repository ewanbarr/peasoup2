include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_io_src
    test_sigproc.cu
    gtest_io.cu
    )

add_definitions(-DPEASOUP_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
cuda_add_executable(gtest_io ${gtest_io_src})
target_link_libraries(gtest_io ${GTEST_LIBRARIES})
add_test(gtest_io gtest_io)
