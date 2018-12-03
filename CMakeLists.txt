cmake_minimum_required(VERSION 3.5)
project(stempy)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

 set(_srcs
  stempy/main.cpp
  stempy/reader.cpp
)

add_executable(stem ${_srcs})
