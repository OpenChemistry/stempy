#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "pyreader.h"
#include <stempy/electron.h>
#include <stempy/electronthresholds.h>
#include <stempy/image.h>
#include <stempy/reader.h>

namespace py = pybind11;

using std::vector;
using namespace stempy;

// These are defined here so we don't leak the pybind11 types into the stemp lib
// we may want to rethink in this is in the future.
namespace stempy {

template <typename BlockType>
CalculateThresholdsResults calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds(blocks, darkReference.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma);
}

template CalculateThresholdsResults calculateThresholds(
  std::vector<Block>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);
template CalculateThresholdsResults calculateThresholds(
  std::vector<PyBlock>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  py::array_t<double> darkReference,
                                  double backgroundThreshold,
                                  double xRayThreshold, int scanWidth = 0,
                                  int scanHeight = 0)
{
  return electronCount(first, last, darkReference.data(), backgroundThreshold,
                       xRayThreshold, scanWidth, scanHeight);
}

} // namespace stempy

PYBIND11_MODULE(_image, m)
{
  py::class_<Image<uint64_t>>(m , "_image_uint64", py::buffer_protocol())
    .def_buffer([](Image<uint64_t>& i) {
       return py::buffer_info(
          i.data.get(),                                                 /* Pointer to buffer */
          sizeof(uint64_t),                                             /* Size of one scalar */
          py::format_descriptor<uint64_t>::format(),                    /* Python struct-style format descriptor */
          2,                                                            /* Number of dimensions */
          { i.height, i.width },  /* Buffer dimensions */
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
          { i.height, i.width},  /* Buffer dimensions */
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
          { r.radii, r.height, r.width},  /* Buffer dimensions */
          { sizeof(uint64_t) * r.width * r.height,                           /* Strides (in bytes) for each index */
            sizeof(uint64_t) * r.width,
            sizeof(uint64_t) });
    });

  py::class_<Image<uint16_t>>(m, "_image_uint16", py::buffer_protocol())
    .def_buffer([](Image<uint16_t>& i) {
      return py::buffer_info(
        i.data.get(),     /* Pointer to buffer */
        sizeof(uint16_t), /* Size of one scalar */
        py::format_descriptor<
          uint16_t>::format(), /* Python struct-style format descriptor */
        2,                     /* Number of dimensions */
        { i.height, i.width }, /* Buffer dimensions */
        { sizeof(uint16_t) * i.width, /* Strides (in bytes) for each index */
          sizeof(uint16_t) });
    });

  py::class_<CalculateThresholdsResults>(m, "_calculate_thresholds_results",
                                         py::buffer_protocol())
    .def_readonly("background_threshold",
                  &CalculateThresholdsResults::backgroundThreshold)
    .def_readonly("xray_threshold", &CalculateThresholdsResults::xRayThreshold)
    .def_readonly("number_of_samples",
                  &CalculateThresholdsResults::numberOfSamples)
    .def_readonly("min_sample", &CalculateThresholdsResults::minSample)
    .def_readonly("max_sample", &CalculateThresholdsResults::maxSample)
    .def_readonly("mean", &CalculateThresholdsResults::mean)
    .def_readonly("variance", &CalculateThresholdsResults::variance)
    .def_readonly("std_dev", &CalculateThresholdsResults::stdDev)
    .def_readonly("number_of_bins", &CalculateThresholdsResults::numberOfBins)
    .def_readonly("xray_threshold_n_sigma",
                  &CalculateThresholdsResults::xRayThresholdNSigma)
    .def_readonly("background_threshold_n_sigma",
                  &CalculateThresholdsResults::backgroundThresholdNSigma);

  py::class_<ElectronCountedData>(m, "_electron_counted_data",
                                  py::buffer_protocol())
    .def_readonly("data", &ElectronCountedData::data)
    .def_readonly("scan_width", &ElectronCountedData::scanWidth)
    .def_readonly("scan_height", &ElectronCountedData::scanHeight)
    .def_readonly("frame_width", &ElectronCountedData::frameWidth)
    .def_readonly("frame_height", &ElectronCountedData::frameHeight);

  // Add more template instantiations as we add more types of iterators
  m.def("create_stem_images", &createSTEMImages<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images", &createSTEMImages<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "create_stem_images_sparse",
    (vector<STEMImage>(*)(const vector<vector<uint32_t>>&, const vector<int>&,
                          const vector<int>&, int, int, int, int, int, int,
                          int)) &
      createSTEMImagesSparse,
    py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images_sparse",
        (vector<STEMImage>(*)(const ElectronCountedData&, const vector<int>&,
                              const vector<int>&, int, int)) &
          createSTEMImagesSparse,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                                Image<double>&, double, double, int, int)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(SectorStreamReader::iterator,
                                SectorStreamReader::iterator, Image<double>&,
                                double, double, int, int)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                                Image<double>&, double, double, int, int)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "electron_count",
    (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                            py::array_t<double>, double, double, int, int)) &
      electronCount,
    py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator,
          py::array_t<double>, double, double, int, int)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "electron_count",
    (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                            py::array_t<double>, double, double, int, int)) &
      electronCount,
    py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults(*)(vector<Block>&, Image<double>&, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults(*)(vector<PyBlock>&, Image<double>&, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults(*)(vector<Block>&, py::array_t<double>, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults(*)(vector<PyBlock>&, py::array_t<double>,
                                       int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("radial_sum", &radialSum<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("radial_sum", &radialSum<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("radial_sum", &radialSum<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("get_container", &getContainer,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_histogram", &createSTEMHistogram,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images", &createSTEMImages<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(StreamReader::iterator, StreamReader::iterator,
                          const Image<double>&)) &
          maximumDiffractionPattern<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(StreamReader::iterator, StreamReader::iterator)) &
          maximumDiffractionPattern<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(SectorStreamReader::iterator,
                          SectorStreamReader::iterator, const Image<double>&)) &
          maximumDiffractionPattern<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(SectorStreamReader::iterator,
                          SectorStreamReader::iterator)) &
          maximumDiffractionPattern<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(PyReader::iterator,
                          PyReader::iterator)) &
          maximumDiffractionPattern<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
}
