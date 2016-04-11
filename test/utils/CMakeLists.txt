include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIBRARY_DIR})

set(
    gtest_logging_src
    test_logging.cpp
    )

add_executable(gtest_logging ${gtest_logging_src})
target_link_libraries(gtest_logging ${GTEST_LIBRARIES})
