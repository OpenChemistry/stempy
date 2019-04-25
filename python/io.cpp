#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/reader.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_io, m)
{
  py::class_<Header>(m, "_header")
    .def_readonly("images_in_block", &Header::imagesInBlock)
    .def_readonly("rows", &Header::rows)
    .def_readonly("columns", &Header::columns)
    .def_readonly("version", &Header::version)
    .def_readonly("timestamp", &Header::timestamp)
    .def_readonly("image_numbers", &Header::imageNumbers);

  py::class_<Block>(m , "_block", py::buffer_protocol())
    .def_readonly("header", &Block::header)
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

  py::class_<StreamReader::iterator>(m, "_reader_iterator")
    .def(py::init<StreamReader*>());

  py::class_<StreamReader>(m, "_reader")
    .def(py::init<const std::string&, uint8_t>())
    .def(py::init<const std::vector<std::string>&, uint8_t>())
    .def("read", (Block(StreamReader::*)()) & StreamReader::read)
    .def("rewind", &StreamReader::rewind)
    .def("begin",
         (StreamReader::iterator(StreamReader::*)()) & StreamReader::begin)
    .def("end", (StreamReader::iterator(StreamReader::*)()) & StreamReader::end)
    .def("process", &StreamReader::process, "", py::arg("stream_id"),
         py::arg("concurrency") = -1, py::arg("width") = 160,
         py::arg("height") = 160, py::arg("url") = "http://127.0.0.1:5000");
}
