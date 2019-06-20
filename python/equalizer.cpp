#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <stempy/image.h>
#include <stempy/equalizer.h>

namespace py = pybind11;

using namespace stempy;

PYBIND11_MODULE(_image, m)
{
   m.def("create_stem_images_histograms", &createSTEMHistograms); 
}