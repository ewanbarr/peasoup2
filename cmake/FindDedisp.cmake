# - Find DEDISP
# Find the native DEDISP includes and library
#
#  DEDISP_INCLUDES    - where to find dedisp.h
#  DEDISP_LIBRARIES   - List of libraries when using DEDISP.
#  DEDISP_FOUND       - True if DEDISP found.

if (DEDISP_INCLUDES)
  # Already in cache, be silent
  set (DEDISP_FIND_QUIETLY TRUE)
endif (DEDISP_INCLUDES)

set(DEDISP_ROOT ${CMAKE_SOURCE_DIR}/thirdparty/dedisp)
find_path (DEDISP_INCLUDES dedisp.h PATH ${DEDISP_ROOT}/include NO_SYSTEM_ENVIRONMENT_PATH)
find_library (DEDISP_LIBRARIES NAMES dedisp PATHS ${DEDISP_ROOT}/lib NO_SYSTEM_ENVIRONMENT_PATH)

#add_library(dedisp SHARED IMPORTED)
#set_target_properties(dedisp PROPERTIES IMPORTED_LOCATION ${DEDISP_ROOT}/lib/libdedisp.so)


# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (DEDISP DEFAULT_MSG DEDISP_LIBRARIES DEDISP_INCLUDES)

mark_as_advanced (DEDISP_LIBRARIES DEDISP_INCLUDES)