#
# Compiler defaults for cheetah
#

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
        message(FATAL_ERROR "GCC version must be at least 4.8!")
    endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "release")
endif ()

if(CMAKE_CXX_COMPILER MATCHES icpc)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcheck -wd2259 -wd1125")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Werror")
    if(CMAKE_BUILD_TYPE MATCHES profile)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0  -fprofile-arcs -ftest-coverage")
    endif()
endif ()
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Werror -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
    if(CMAKE_BUILD_TYPE MATCHES profile)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0  -fprofile-arcs -ftest-coverage")
    endif()
endif ()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wextra -pedantic ")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic ")
