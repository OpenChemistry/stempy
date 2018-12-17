cmake_minimum_required(VERSION 3.5)
project(stempy)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(InstallLocation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(SYSTEM
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ThreadPool
)

find_package(Threads)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(stem
  stempy/reader.cpp
  stempy/mask.cpp
  stempy/image.cpp
)

target_link_libraries(stem
  PRIVATE Threads::Threads)

add_executable(stemcli
  stempy/main.cpp
)

target_link_libraries(stemcli
  PRIVATE stem)

set_target_properties(stem
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/pybind11)
add_subdirectory(python)
