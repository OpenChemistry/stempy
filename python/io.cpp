#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/reader.h>
#include <python/reader_h5.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_io, m)
{
  py::class_<Header>(m, "_header")
    .def_readonly("images_in_block", &Header::imagesInBlock)
    .def_readonly("frame_height", &Header::frameHeight)
    .def_readonly("frame_width", &Header::frameWidth)
    .def_readonly("version", &Header::version)
    .def_readonly("timestamp", &Header::timestamp)
    .def_readonly("image_numbers", &Header::imageNumbers)
    .def_readonly("scan_height", &Header::scanHeight)
    .def_readonly("scan_width", &Header::scanWidth);

  py::class_<Block>(m, "_block", py::buffer_protocol())
    .def_readonly("header", &Block::header)
    .def_buffer([](Block& b) {
      return py::buffer_info(
        b.data.get(),     /* Pointer to buffer */
        sizeof(uint16_t), /* Size of one scalar */
        py::format_descriptor<
          uint16_t>::format(), /* Python struct-style format descriptor */
        3,                     /* Number of dimensions */
        { b.header.imagesInBlock, b.header.frameHeight,
          b.header.frameWidth }, /* Buffer dimensions */
        { sizeof(uint16_t) * b.header.frameHeight * b.header.frameWidth,
          sizeof(uint16_t) *
            b.header.frameHeight, /* Strides (in bytes) for each index */
          sizeof(uint16_t) });
    });

  py::class_<StreamReader::iterator>(m, "_reader_iterator")
    .def(py::init<StreamReader*>());

  py::class_<StreamReader>(m, "_reader")
    .def(py::init<const std::string&, uint8_t>())
    .def(py::init<const std::vector<std::string>&, uint8_t>())
    .def("read", (Block(StreamReader::*)()) & StreamReader::read)
    .def("reset", &StreamReader::reset)
    .def("begin",
         (StreamReader::iterator(StreamReader::*)()) & StreamReader::begin)
    .def("end", (StreamReader::iterator(StreamReader::*)()) & StreamReader::end)
    .def("process", &StreamReader::process, "", py::arg("stream_id"),
         py::arg("concurrency") = -1, py::arg("width") = 160,
         py::arg("height") = 160, py::arg("url") = "http://127.0.0.1:5000");
   

  py::class_<H5Reader::iterator>(m, "_h5reader_iterator")
    .def(py::init<H5Reader*>());

  py::class_<H5Reader>(m, "_h5reader")
    .def(py::init<py::object, std::vector<uint32_t>&, uint32_t, uint32_t, uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>())
    .def("read", (PyBlock(H5Reader::*)()) & H5Reader::read)
    .def("begin",(H5Reader::iterator(H5Reader::*)()) & H5Reader::begin)
    .def("end", (H5Reader::iterator(H5Reader::*)()) & H5Reader::end);
  
}
