#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <python/pyreader.h>
#include <stempy/reader.h>

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
    .def("end",
         (StreamReader::iterator(StreamReader::*)()) & StreamReader::end);

  py::class_<PyReader::iterator>(m, "_pyreader_iterator")
    .def(py::init<PyReader*>());

  py::class_<PyReader>(m, "_pyreader")
    .def(py::init<py::object, std::vector<uint32_t>&, uint32_t, uint32_t,
                  uint32_t, uint32_t>())
    .def("read", (PyBlock(PyReader::*)()) & PyReader::read)
    .def("begin", (PyReader::iterator(PyReader::*)()) & PyReader::begin)
    .def("end", (PyReader::iterator(PyReader::*)()) & PyReader::end);

  py::class_<SectorStreamReader::iterator>(m, "_sector_reader_iterator")
    .def(py::init<SectorStreamReader*>());

  py::class_<SectorStreamReader>(m, "_sector_reader")
    .def(py::init<const std::string&>())
    .def(py::init<const std::vector<std::string>&>())
    .def("read", (Block(SectorStreamReader::*)()) & SectorStreamReader::read)
    .def("reset", &SectorStreamReader::reset)
    .def("begin", (SectorStreamReader::iterator(SectorStreamReader::*)()) &
                    SectorStreamReader::begin)
    .def("end", (SectorStreamReader::iterator(SectorStreamReader::*)()) &
                  SectorStreamReader::end)
    .def("data_captured", &SectorStreamReader::dataCaptured)
    .def("to_hdf5", &SectorStreamReader::toHdf5);
}
