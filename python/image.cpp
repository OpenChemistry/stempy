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
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds(blocks, darkReference.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma);
}

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<double> darkReference,
  py::array_t<float> gain,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds(blocks, darkReference.data(), gain.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma);
}

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, Image<double>& darkReference,
  py::array_t<float> gain, int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds(blocks, darkReference.data.get(), gain.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma);
}

template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<Block>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);
template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<PyBlock>& blocks, py::array_t<double> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);

template CalculateThresholdsResults<float> calculateThresholds(
  std::vector<Block>& blocks, py::array_t<double> darkReference,
  py::array_t<float> gain, int numberOfSamples,
  double backgroundThresholdNSigma,  double xRayThresholdNSigma);
template CalculateThresholdsResults<float> calculateThresholds(
  std::vector<PyBlock>& blocks, py::array_t<double> darkReference,
  py::array_t<float> gain,  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  py::array_t<double> darkReference,
                                  py::array_t<float> gain,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = {0, 0})
{
  return electronCount(first, last, darkReference.data(), gain.data(), backgroundThreshold,
                       xRayThreshold, scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<double>& darkReference,
                                  py::array_t<float> gain,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = {0, 0})
{
  return electronCount(first, last, darkReference, gain.data(), backgroundThreshold,
                       xRayThreshold, scanDimensions);
}

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, py::array_t<double> darkReference,
  py::array_t<float> gain, int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false)
{
  return electronCount(reader, darkReference.data(), gain.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, scanDimensions, verbose);
}

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, Image<double>& darkReference,
  py::array_t<float> gain, int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false)
{
  return electronCount(reader, darkReference, gain.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, scanDimensions, verbose);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  py::array_t<double> darkReference,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = {0, 0})
{
  return electronCount(first, last, darkReference.data(), backgroundThreshold,
                       xRayThreshold, scanDimensions);
}

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, py::array_t<double> darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false)
{
  return electronCount(reader, darkReference.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, scanDimensions, verbose);
}

} // namespace stempy

PYBIND11_MODULE(_image, m)
{
  py::class_<Image<uint64_t>>(m, "_image_uint64", py::buffer_protocol())
    .def_buffer([](Image<uint64_t>& i) {
      return py::buffer_info(
        i.data.get(),                              /* Pointer to buffer */
        sizeof(uint64_t),                          /* Size of one scalar */
        py::format_descriptor<uint64_t>::format(), /* Python struct-style format
                                                      descriptor */
        2,                                         /* Number of dimensions */
        { i.dimensions.second, i.dimensions.first }, /* Buffer dimensions */
        { sizeof(uint64_t) *
            i.dimensions.first, /* Strides (in bytes) for each index */
          sizeof(uint64_t) });
    });

  py::class_<Image<double>>(m, "_image_double", py::buffer_protocol())
    .def_buffer([](Image<double>& i) {
      return py::buffer_info(
        i.data.get(),                            /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format
                                                    descriptor */
        2,                                       /* Number of dimensions */
        { i.dimensions.second, i.dimensions.first }, /* Buffer dimensions */
        { sizeof(double) *
            i.dimensions.first, /* Strides (in bytes) for each index */
          sizeof(double) });
    });

  py::class_<RadialSum<uint64_t>>(m, "_radial_sum_uint64",
                                  py::buffer_protocol())
    .def_buffer([](RadialSum<uint64_t>& r) {
      return py::buffer_info(
        r.data.get(),                              /* Pointer to buffer */
        sizeof(uint64_t),                          /* Size of one scalar */
        py::format_descriptor<uint64_t>::format(), /* Python struct-style format
                                                      descriptor */
        3,                                         /* Number of dimensions */
        { r.radii, r.dimensions.second,
          r.dimensions.first }, /* Buffer dimensions */
        { sizeof(uint64_t) * r.dimensions.first *
            r.dimensions.second, /* Strides (in bytes) for each index */
          sizeof(uint64_t) * r.dimensions.first, sizeof(uint64_t) });
    });

  py::class_<Image<uint16_t>>(m, "_image_uint16", py::buffer_protocol())
    .def_buffer([](Image<uint16_t>& i) {
      return py::buffer_info(
        i.data.get(),                              /* Pointer to buffer */
        sizeof(uint16_t),                          /* Size of one scalar */
        py::format_descriptor<uint16_t>::format(), /* Python struct-style format
                                                      descriptor */
        2,                                         /* Number of dimensions */
        { i.dimensions.second, i.dimensions.first }, /* Buffer dimensions */
        { sizeof(uint16_t) *
            i.dimensions.first, /* Strides (in bytes) for each index */
          sizeof(uint16_t) });
    });

  py::class_<CalculateThresholdsResults<uint16_t>>(m, "_calculate_thresholds_results",
                                         py::buffer_protocol())
    .def_readonly("background_threshold",
                  &CalculateThresholdsResults<uint16_t>::backgroundThreshold)
    .def_readonly("xray_threshold", &CalculateThresholdsResults<uint16_t>::xRayThreshold)
    .def_readonly("number_of_samples",
                  &CalculateThresholdsResults<uint16_t>::numberOfSamples)
    .def_readonly("min_sample", &CalculateThresholdsResults<uint16_t>::minSample)
    .def_readonly("max_sample", &CalculateThresholdsResults<uint16_t>::maxSample)
    .def_readonly("mean", &CalculateThresholdsResults<uint16_t>::mean)
    .def_readonly("variance", &CalculateThresholdsResults<uint16_t>::variance)
    .def_readonly("std_dev", &CalculateThresholdsResults<uint16_t>::stdDev)
    .def_readonly("number_of_bins", &CalculateThresholdsResults<uint16_t>::numberOfBins)
    .def_readonly("xray_threshold_n_sigma",
                  &CalculateThresholdsResults<uint16_t>::xRayThresholdNSigma)
    .def_readonly("background_threshold_n_sigma",
                  &CalculateThresholdsResults<uint16_t>::backgroundThresholdNSigma)
    .def_readonly("optimized_mean", &CalculateThresholdsResults<uint16_t>::optimizedMean)
    .def_readonly("optimized_std_dev",
                  &CalculateThresholdsResults<uint16_t>::optimizedStdDev);

  py::class_<ElectronCountedData>(m, "_electron_counted_data",
                                  py::buffer_protocol())
    .def_readonly("data", &ElectronCountedData::data)
    .def_readonly("scan_dimensions", &ElectronCountedData::scanDimensions)
    .def_readonly("frame_dimensions", &ElectronCountedData::frameDimensions);

  // Add more template instantiations as we add more types of iterators
  m.def("create_stem_images", &createSTEMImages<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images", &createSTEMImages<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images",
        (vector<STEMImage>(*)(const vector<vector<uint32_t>>&,
                              const vector<int>&, const vector<int>&,
                              Dimensions2D, Dimensions2D, Coordinates2D, int)) &
          createSTEMImages,
        py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images",
        (vector<STEMImage>(*)(const ElectronCountedData&, const vector<int>&,
                              const vector<int>&, Coordinates2D)) &
          createSTEMImages,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());

  // Electron counting with gain
  m.def("electron_count",
        (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                                Image<double>&, py::array_t<float>, double,
                                double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(SectorStreamReader::iterator,
                                SectorStreamReader::iterator, Image<double>&,
                                py::array_t<float>, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                                Image<double>&, py::array_t<float>, double,
                                double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                                py::array_t<double>, py::array_t<float>, double,
                                double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator,
          py::array_t<double>, py::array_t<float>, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                                py::array_t<double>, py::array_t<float>, double,
                                double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(SectorStreamThreadedReader*, Image<double>&,
                                py::array_t<float>, int, int, double, double,
                                Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "electron_count",
    (ElectronCountedData(*)(SectorStreamThreadedReader*, py::array_t<double>,
                            py::array_t<float>, int, int, double, double,
                            Dimensions2D, bool)) &
      electronCount,
    py::call_guard<py::gil_scoped_release>());

  // Electron counting, without gain
  m.def("electron_count",
        (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                                Image<double>&, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(SectorStreamReader::iterator,
                                SectorStreamReader::iterator, Image<double>&,
                                double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                                Image<double>&, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(StreamReader::iterator, StreamReader::iterator,
                                py::array_t<double>, double, double,
                                Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator,
          py::array_t<double>, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(PyReader::iterator, PyReader::iterator,
                                py::array_t<double>, double, double,
                                Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedData(*)(SectorStreamThreadedReader*, Image<double>&,
                                int, int, double, double, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "electron_count",
    (ElectronCountedData(*)(SectorStreamThreadedReader*, py::array_t<double>,
                            int, int, double, double, Dimensions2D, bool)) &
      electronCount,
    py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, with gain
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<Block>&, Image<double>&,
          py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<PyBlock>&, Image<double>&,
          py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<Block>&, py::array_t<double>,
          py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<PyBlock>&, py::array_t<double>,
          py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, without gain
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<Block>&, Image<double>&, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<PyBlock>&, Image<double>&, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<Block>&, py::array_t<double>, int,
                                       double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<PyBlock>&, py::array_t<double>,
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
