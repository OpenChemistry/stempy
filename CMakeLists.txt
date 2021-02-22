cmake_minimum_required(VERSION 3.19)
project(stempy)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(InstallLocation)
include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

include_directories(SYSTEM
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ThreadPool
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

include_directories(SYSTEM
  ${Python3_INCLUDE_DIRS}
)

find_package(HDF5 REQUIRED COMPONENTS C)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(_vtkm_src
  stempy/image.cpp
  stempy/electron.cpp
)
set(_src
  stempy/reader.cpp
  python/pyreader.cpp
  stempy/mask.cpp
  stempy/electronthresholds.cpp
  h5cpp/h5readwrite.cpp)

set(_libs Threads::Threads)
list(APPEND _libs hdf5::hdf5)
list(APPEND _libs pybind11::headers)
list(APPEND _libs pybind11::python_link_helper)

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

set(VTKm ${VTKm_FOUND})
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

if(WIN32)
  target_link_libraries(stem PRIVATE ${Python3_LIBRARIES})
endif()

add_subdirectory(python)
