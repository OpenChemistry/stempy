#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/reader.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_io, m)
{
  py::class_<Header>(m, "_header")
    .def_readwrite("images_in_block", &Header::imagesInBlock)
    .def_readwrite("rows", &Header::rows)
    .def_readwrite("columns", &Header::columns)
    .def_readwrite("version", &Header::version)
    .def_readwrite("timestamp", &Header::timestamp)
    .def_readwrite("image_numbers", &Header::imageNumbers);

  py::class_<Block>(m , "_block", py::buffer_protocol())
    .def_readwrite("header", &Block::header)
    .def_buffer([](Block& b) {
       return py::buffer_info(
          b.data.get(),                                                 /* Pointer to buffer */
          sizeof(uint16_t),                                             /* Size of one scalar */
          py::format_descriptor<uint16_t>::format(),                    /* Python struct-style format descriptor */
          3,                                                            /* Number of dimensions */
          { b.header.imagesInBlock, b.header.rows, b.header.columns },  /* Buffer dimensions */
          { sizeof(uint16_t) * b.header.rows * b.header.columns,
            sizeof(uint16_t) * b.header.rows,                           /* Strides (in bytes) for each index */
            sizeof(uint16_t) });
    });

  py::class_<StreamReader>(m, "_reader")
    .def(py::init<const std::string &>())
    .def("read", (Block (StreamReader::*)())&StreamReader::read);
}
