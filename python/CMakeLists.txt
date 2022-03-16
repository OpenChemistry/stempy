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

if(NOT WIN32)
  set_target_properties(_io _image PROPERTIES
                        BUILD_RPATH ${_rpath_value}
                        INSTALL_RPATH ${_rpath_value})
endif()

# Install all python files
set(python_dir "${CMAKE_CURRENT_SOURCE_DIR}/stempy")
file(GLOB_RECURSE python_files RELATIVE "${python_dir}" "*.py")
foreach(file ${python_files})
  set(destination_file "${_python_module_install_dir}/${file}")
  get_filename_component(destination "${destination_file}" DIRECTORY)
  install(FILES "stempy/${file}" COMPONENT python DESTINATION "${destination}")
endforeach()

if(NOT SKBUILD)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/stempy/")

  # Create soft links to all root files and directories in stempy/python/stempy
  set(exclude_list "__pycache__")
  file(GLOB link_paths RELATIVE "${python_dir}" "${python_dir}/*")
  foreach(path ${link_paths})
    set(full_path "${python_dir}/${path}")
    if ("${path}" IN_LIST exclude_list)
      continue()
    endif()

    # If it is a file but doesn't end with .py, exclude it
    if (NOT IS_DIRECTORY "${full_path}" AND NOT "${path}" MATCHES ".py$")
      continue()
    endif()

    set(dest_dir "${CMAKE_BINARY_DIR}/lib/stempy")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${full_path}" "${dest_dir}/${path}")
  endforeach()
endif()
