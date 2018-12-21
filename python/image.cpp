#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/image.h>
#include <stempy/reader.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_image, m)
{
  py::class_<Image>(m , "_image", py::buffer_protocol())
    .def_buffer([](Image& i) {
       return py::buffer_info(
          i.data.get(),                                                 /* Pointer to buffer */
          sizeof(uint64_t),                                             /* Size of one scalar */
          py::format_descriptor<uint64_t>::format(),                    /* Python struct-style format descriptor */
          2,                                                            /* Number of dimensions */
          { i.width, i.height },  /* Buffer dimensions */
          { sizeof(uint64_t) * i.width,                           /* Strides (in bytes) for each index */
            sizeof(uint64_t) });
    });

  py::class_<STEMImage>(m, "_stem_image")
    .def_readonly("bright", &STEMImage::bright)
    .def_readonly("dark", &STEMImage::dark);

  m.def("create_stem_image", &createSTEMImage);

}
