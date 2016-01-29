if (NVTX_INCLUDES)
  # Already in cache, be silent
  set (NVTX_FIND_QUIETLY TRUE)
endif (NVTX_INCLUDES)

find_path (NVTX_INCLUDES nvToolsExt.h PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include)
find_library (NVTX_LIBRARIES NAMES nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# handle the QUIETLY and REQUIRED arguments and set NVTX_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NVTX DEFAULT_MSG NVTX_LIBRARIES NVTX_INCLUDES)

mark_as_advanced (NVTX_LIBRARIES NVTX_INCLUDES)