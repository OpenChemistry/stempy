cmake_minimum_required(VERSION 3.5)
project(stempy)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(InstallLocation)

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
find_package(SocketIOClientCpp)
include_directories(
  ${SocketIOClientCpp_INCLUDE_DIRS}
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)


set(_vtkm_src
  stempy/image.cpp
)
set(_src stempy/reader.cpp
  stempy/mask.cpp)

if(SocketIOClientCpp_FOUND)
  list(APPEND _src stempy/sioclient.cpp)
endif()

set(_libs Threads::Threads)

find_package(VTKm COMPONENTS OpenMP)
if(VTKm_FOUND)
  list(APPEND _libs vtkm_cont)
endif()

if(TARGET vtkm::cuda)
  # Compile with CUDA
  vtkm_compile_as_cuda(cudaSource ${_vtkm_src})
  add_library(stem ${cudaSource} ${_src})
else()
  add_library(stem ${_src} ${_vtkm_src})
endif()

set_property(TARGET stem PROPERTY POSITION_INDEPENDENT_CODE ON)

if(SocketIOClientCpp_FOUND)
  list(APPEND _libs sioclient)
endif()

target_link_libraries(stem
  PRIVATE ${_libs})

add_executable(stemcli stempy/main.cpp)

target_link_libraries(stemcli
  PRIVATE stem)

set_target_properties(stem
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

set(SocketIOClientCpp ${SocketIOClientCpp_FOUND})
set(VTKm ${VTKm_FOUND})
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11)
add_subdirectory(python)
