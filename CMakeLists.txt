cmake_minimum_required(VERSION 3.19)
project(stempy)

set(stempy_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${stempy_CMAKE_DIR})

include(InstallLocation)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

if(WIN32 AND SKBUILD)
  # For building with setup.py, we want everything in stempy
  set(CMAKE_INSTALL_BINDIR "stempy")
  set(CMAKE_INSTALL_LIBDIR "stempy")
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(Threads)

include_directories(
  SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/least-squares-cpp/include
)

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11)
set(PYBIND11_PYTHON_VERSION "3" CACHE STRING "")
set(PYBIND11_CPP_STANDARD "-std=c++14" CACHE STRING "")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(_libs Threads::Threads)

option(stempy_ENABLE_MPI "Build with MPI" OFF)
if (stempy_ENABLE_MPI)
  find_package(MPI REQUIRED)
  list(APPEND _libs MPI::MPI_C)

  include_directories(
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cereal/include
  )

  include_directories(
    SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Empirical/include
  )

  find_package(BigMPI REQUIRED)
  list(APPEND _libs bigmpi::bigmpi)
endif()
set(USE_MPI ${MPI_FOUND})

set(_vtkm_src
  stempy/image.cpp
  stempy/electron.cpp
)
set(_src
  stempy/reader.cpp
  python/pyreader.cpp
  stempy/mask.cpp
  stempy/electronthresholds.cpp)

if (stempy_ENABLE_MPI)
  list(APPEND _src
    stempy/electron_mpi.cpp
  )
  list(APPEND _src
    stempy/reader_mpi.cpp
  )
endif()

list(APPEND _libs pybind11::headers)
# We have to link to Python3::Module since "pyreader.cpp" includes
# pybind11 and python header files.
list(APPEND _libs Python3::Module)

option(stempy_ENABLE_HDF5 "Build with HDF5" ON)
if (stempy_ENABLE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C)
  list(APPEND _src h5cpp/h5readwrite.cpp)
  list(APPEND _libs hdf5::hdf5)
endif()

option(stempy_ENABLE_VTKm "Build with VTK-m" OFF)
cmake_dependent_option(stempy_ENABLE_CUDA "Enable VTK-m CUDA backend" OFF "stempy_ENABLE_VTKm" ON)
cmake_dependent_option(stempy_ENABLE_OPENMP "Build VTK-m OpenMP backend" ON "stempy_ENABLE_VTKm" ON)
if (stempy_ENABLE_VTKm)
  set(_components "")
  if(stempy_ENABLE_CUDA)
      list(APPEND _components "CUDA")
  endif()
  if(stempy_ENABLE_OPENMP)
      list(APPEND _components "OpenMP")
  endif()
  find_package(VTKm COMPONENTS "${_components}" REQUIRED)
  list(APPEND _libs vtkm_cont)
endif()

if(stempy_ENABLE_CUDA AND TARGET vtkm::cuda)
  # Compile with CUDA
  vtkm_compile_as_cuda(cudaSource ${_vtkm_src})
  add_library(stem ${cudaSource} ${_src})
else()
  add_library(stem ${_src} ${_vtkm_src})
endif()

# Version 1.12 of HDF5 deprecates the get_info_by_name1(...) interface and
# the get_info_by_name maps to get_info_by_name3(...) which has a different
# signature. We pass these compability flags so that we can continue to
# use the older signature with newer version of HDF5.
if (stempy_ENABLE_HDF5 AND NOT ("${HDF5_VERSION}" VERSION_LESS 1.12.0))
  target_compile_options(stem PRIVATE "-DH5Oget_info_by_name_vers=1")
  target_compile_options(stem PRIVATE "-DH5O_info_t_vers=1")
  target_compile_options(stem PRIVATE "-DH5Ovisit_vers=1")
endif()

target_include_directories(stem
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ThreadPool>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR};${CMAKE_INSTALL_INCLUDEDIR}/thirdparty/ThreadPool>"
)

set_property(TARGET stem PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(stem
  PRIVATE ${_libs})

set_target_properties(stem
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

# Set the RPATH so the dependent libraries can be found in the wheel.
if(APPLE)
  set(_rpath_value "@loader_path")
elseif(UNIX)
  set(_rpath_value "$ORIGIN")
endif()

if (NOT WIN32)
  set_target_properties(stem PROPERTIES
                        INSTALL_RPATH ${_rpath_value})
endif()

set(_python_module_install_dir "stempy")
# SKBUILD is set for binary wheel
if (NOT SKBUILD)
  set(_python_module_install_dir "${INSTALL_LIBRARY_DIR}/stempy")
endif()
install(TARGETS stem LIBRARY DESTINATION "${_python_module_install_dir}")

install(TARGETS stem
        EXPORT stempyTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES stempy/reader.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stempy)
install(FILES stempy/image.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stempy)
install(FILES stempy/mask.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stempy)
install(FILES stempy/electron.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stempy)
install(FILES thirdparty/ThreadPool/ThreadPool.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/thirdparty/ThreadPool)

set(VTKm ${VTKm_FOUND})
set(ENABLE_HDF5 ${HDF5_FOUND})
configure_file("${stempy_CMAKE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stempy)
install(EXPORT stempyTargets
      FILE stempyTargets.cmake
      NAMESPACE stempy::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stempy
)

# Also run export(...) so we can use from a build directory as well
export(TARGETS stem NAMESPACE stempy:: FILE stempyTargets.cmake)

configure_package_config_file(${stempy_CMAKE_DIR}/stempyConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/stempyConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stempy
)

install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/stempyConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stempy
)

add_subdirectory(python)
