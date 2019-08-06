cmake_minimum_required(VERSION 3.5)
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
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
set(PYBIND11_PYTHON_VERSION "3" CACHE STRING "")
set(PYBIND11_CPP_STANDARD "-std=c++14" CACHE STRING "")

include_directories(SYSTEM
  ${PYTHON_INCLUDE_PATH}
  ${PYBIND11_INCLUDE_DIR}
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(_vtkm_src
  stempy/image.cpp
  stempy/electron.cpp
)
set(_src stempy/reader.cpp python/pyreader.cpp
  stempy/mask.cpp
  stempy/electronthresholds.cpp)

set(_libs Threads::Threads)

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

add_executable(stemcli stempy/main.cpp)

target_link_libraries(stemcli
  PRIVATE stem pybind11::embed)

set_target_properties(stem
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

set(VTKm ${VTKm_FOUND})
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

add_subdirectory(python)
