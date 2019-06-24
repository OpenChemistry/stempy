#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/electron.h>
#include <stempy/electronthresholds.h>
#include <stempy/image.h>
#include <stempy/reader.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_image, m)
{
  py::class_<Image<uint64_t>>(m , "_image_uint64", py::buffer_protocol())
    .def_buffer([](Image<uint64_t>& i) {
       return py::buffer_info(
          i.data.get(),                                                 /* Pointer to buffer */
          sizeof(uint64_t),                                             /* Size of one scalar */
          py::format_descriptor<uint64_t>::format(),                    /* Python struct-style format descriptor */
          2,                                                            /* Number of dimensions */
          { i.width, i.height },  /* Buffer dimensions */
          { sizeof(uint64_t) * i.width,                           /* Strides (in bytes) for each index */
            sizeof(uint64_t) });
    });

  py::class_<Image<double>>(m , "_image_double", py::buffer_protocol())
    .def_buffer([](Image<double>& i) {
       return py::buffer_info(
          i.data.get(),                                                 /* Pointer to buffer */
          sizeof(double),                                               /* Size of one scalar */
          py::format_descriptor<double>::format(),                    /* Python struct-style format descriptor */
          2,                                                            /* Number of dimensions */
          { i.width, i.height },  /* Buffer dimensions */
          { sizeof(double) * i.width,                           /* Strides (in bytes) for each index */
            sizeof(double) });
    });

  py::class_<RadialSum<uint64_t>>(m , "_radial_sum_uint64", py::buffer_protocol())
    .def_buffer([](RadialSum<uint64_t>& r) {
       return py::buffer_info(
          r.data.get(),                                                 /* Pointer to buffer */
          sizeof(uint64_t),                                               /* Size of one scalar */
          py::format_descriptor<uint64_t>::format(),                    /* Python struct-style format descriptor */
          3,                                                            /* Number of dimensions */
          { r.radii, r.width, r.height},  /* Buffer dimensions */
          { sizeof(uint64_t) * r.width * r.height,                           /* Strides (in bytes) for each index */
            sizeof(uint64_t) * r.width,
            sizeof(uint64_t) });
    });


  // Add more template instantiations as we add more types of iterators
  m.def("create_stem_images", &createSTEMImages<StreamReader::iterator>);
  m.def("create_stem_images_sparse", &createSTEMImagesSparse);
  m.def("calculate_average", &calculateAverage<StreamReader::iterator>);
  m.def("electron_count", &electronCount<StreamReader::iterator>);
  m.def("calculate_thresholds", &calculateThresholds);
  m.def("radial_sum", &radialSum<StreamReader::iterator>);

  m.def("get_container", &getContainer);
  m.def("create_stem_histogram", &createSTEMHistogram);
}
