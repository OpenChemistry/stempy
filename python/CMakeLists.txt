pybind11_add_module(_io io.cpp pyreader.cpp)
pybind11_add_module(_image image.cpp pyreader.cpp)

target_link_libraries(_io
  PRIVATE
    pybind11::module
    stem
)

target_link_libraries(_image
  PRIVATE
    pybind11::module
    stem
)

install(TARGETS _io LIBRARY COMPONENT python DESTINATION "${_python_module_install_dir}")
install(TARGETS _image LIBRARY COMPONENT python DESTINATION "${_python_module_install_dir}")

set_target_properties(_io _image
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy"
)

if (NOT WIN32)
  set_target_properties(_io _image PROPERTIES
                        INSTALL_RPATH ${_rpath_value})
endif()

install(FILES stempy/io/__init__.py  COMPONENT python DESTINATION "${_python_module_install_dir}/io")
install(FILES stempy/image/__init__.py  COMPONENT python DESTINATION "${_python_module_install_dir}/image")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy/")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/stempy/io" "${CMAKE_BINARY_DIR}/lib/stempy/io")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/stempy/image" "${CMAKE_BINARY_DIR}/lib/stempy/image")
