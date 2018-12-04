find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
set(PYBIND11_PYTHON_VERSION "3" CACHE STRING "")
set(PYBIND11_CPP_STANDARD "-std=c++14" CACHE STRING "")

include_directories(SYSTEM
  ${PYTHON_INCLUDE_PATH}
  ${pybind11_INCLUDE_DIRS}
  )

pybind11_add_module(_io io.cpp)

target_link_libraries(_io
  PRIVATE
    pybind11::module
    reader
)

install(TARGETS _io LIBRARY COMPONENT python DESTINATION "${INSTALL_LIBRARY_DIR}/stempy/")

set_target_properties(_io
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

install(FILES stempy/io/__init__.py  COMPONENT python DESTINATION "${INSTALL_LIBRARY_DIR}/stempy/io/")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy/")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/stempy/io" "${CMAKE_BINARY_DIR}/lib/stempy/io")