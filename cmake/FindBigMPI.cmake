# Defines:
#
#  BIGMPI_FOUND        - system has
#  BIGMPI_INCLUDE_DIRS - the BigMPI include directories
#  BIGMPI_LIBRARY      - the BigMPI library
#
find_path(BIGMPI_INCLUDE_DIR bigmpi.h)
find_library(BIGMPI_LIBRARY NAMES bigmpi)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BigMPI DEFAULT_MSG BIGMPI_INCLUDE_DIR
                                  BIGMPI_LIBRARY)

mark_as_advanced(BIGMPI_INCLUDE_DIR BIGMPI_LIBRARY)

if(BIGMPI_FOUND)
  set(BIGMPI_INCLUDE_DIRS "${BIGMPI_INCLUDE_DIR}")

  if(NOT TARGET bigmpi::bigmpi)
    add_library(bigmpi::bigmpi SHARED IMPORTED GLOBAL)
    set_target_properties(bigmpi::bigmpi PROPERTIES
      IMPORTED_LOCATION "${BIGMPI_LIBRARY}"
      IMPORTED_IMPLIB "${BIGMPI_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${BIGMPI_INCLUDE_DIR}")
  endif()
endif()
