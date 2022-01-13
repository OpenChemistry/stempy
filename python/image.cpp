#include <pybind11/numpy.h>
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

template <typename T>
py::array_t<T> vectorToPyArray(std::vector<T>&& v)
{
  // Steals the vector pointer and returns a numpy array with the data
  auto deleter = [](void* v) { delete reinterpret_cast<std::vector<T>*>(v); };
  auto* ptr = new std::vector<T>(std::move(v));
  auto capsule = py::capsule(ptr, deleter);
  return py::array(ptr->size(), ptr->data(), capsule);
}

struct ElectronCountedDataPyArray
{

  ElectronCountedDataPyArray(ElectronCountedData&& other)
  {
    data.reserve(other.data.size());
    for (auto& vec : other.data) {
      data.push_back(vectorToPyArray(std::move(vec)));
    }

    scanPositions = other.scanPositions;
    scanDimensions = other.scanDimensions;
    frameDimensions = other.frameDimensions;
  }

  std::vector<py::array_t<uint32_t>> data;
  std::vector<uint32_t> scanPositions;

  Dimensions2D scanDimensions = { 0, 0 };
  Dimensions2D frameDimensions = { 0, 0 };
};

template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds(blocks, darkReference.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma);
}

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain)
{
  return calculateThresholds(blocks, darkReference.data(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma,
                             gain.data());
}

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, Image<float>& darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain)
{
  return calculateThresholds(blocks, darkReference.data.get(), numberOfSamples,
                             backgroundThresholdNSigma, xRayThresholdNSigma,
                             gain.data());
}

// With gain, without darkreference
template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  py::array_t<float> gain)
{

  return calculateThresholds<BlockType, float, false>(
    blocks, nullptr, numberOfSamples, backgroundThresholdNSigma,
    xRayThresholdNSigma, gain.data());
}

// Only dark reference
template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<Block>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);
template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<PyBlock>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);

// Gain and darkreference
template CalculateThresholdsResults<float> calculateThresholds(
  std::vector<Block>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain);
template CalculateThresholdsResults<float> calculateThresholds(
  std::vector<PyBlock>& blocks, py::array_t<float> darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain);

template <typename InputIt>
ElectronCountedDataPyArray electronCount(InputIt first, InputIt last,
                                         py::array_t<float> darkReference,
                                         double backgroundThreshold,
                                         double xRayThreshold,
                                         py::array_t<float> gain,
                                         Dimensions2D scanDimensions = { 0, 0 })
{
  return electronCount(first, last, darkReference.data(), backgroundThreshold,
                       xRayThreshold, gain.data(), scanDimensions);
}

template <typename InputIt>
ElectronCountedDataPyArray electronCount(InputIt first, InputIt last,
                                         Image<float>& darkReference,
                                         double backgroundThreshold,
                                         double xRayThreshold,
                                         py::array_t<float> gain,
                                         Dimensions2D scanDimensions = { 0, 0 })
{
  return electronCount(first, last, darkReference, backgroundThreshold,
                       xRayThreshold, gain.data(), scanDimensions);
}

template <typename InputIt>
ElectronCountedDataPyArray electronCount(InputIt first, InputIt last,
                                         py::array_t<float> darkReference,
                                         double backgroundThreshold,
                                         double xRayThreshold,
                                         Dimensions2D scanDimensions = { 0, 0 })
{
  return electronCount(first, last, darkReference.data(), backgroundThreshold,
                       xRayThreshold, scanDimensions);
}

template <typename InputIt>
ElectronCountedDataPyArray electronCount(InputIt first, InputIt last,
                                         double backgroundThreshold,
                                         double xRayThreshold,
                                         py::array_t<float> gain,
                                         Dimensions2D scanDimensions = { 0, 0 })
{
  return electronCount(first, last, backgroundThreshold, xRayThreshold,
                       gain.data(), scanDimensions);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamThreadedReader* reader, py::array_t<float> darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  py::array_t<float> gain, Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, gain.data(), scanDimensions,
                       verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamThreadedReader* reader, Image<float>& darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  py::array_t<float> gain, Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference, thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, gain.data(), scanDimensions,
                       verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain,
  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, thresholdNumberOfBlocks, numberOfSamples,
                       backgroundThresholdNSigma, xRayThresholdNSigma,
                       gain.data(), scanDimensions, verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamThreadedReader* reader, py::array_t<float> darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, scanDimensions, verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamMultiPassThreadedReader* reader, py::array_t<float> darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  py::array_t<float> gain, Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, gain.data(), scanDimensions,
                       verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamMultiPassThreadedReader* reader, Image<float>& darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  py::array_t<float> gain, Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference, thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, gain.data(), scanDimensions,
                       verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamMultiPassThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, py::array_t<float> gain,
  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, thresholdNumberOfBlocks, numberOfSamples,
                       backgroundThresholdNSigma, xRayThresholdNSigma,
                       gain.data(), scanDimensions, verbose);
}

ElectronCountedDataPyArray electronCount(
  SectorStreamMultiPassThreadedReader* reader, py::array_t<float> darkReference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(reader, darkReference.data(), thresholdNumberOfBlocks,
                       numberOfSamples, backgroundThresholdNSigma,
                       xRayThresholdNSigma, scanDimensions, verbose);
}

// Explicitly instantiate version for py::array_t
template std::vector<STEMImage> createSTEMImages(
  const std::vector<py::array_t<uint32_t>>& sparseData,
  const std::vector<int>& innerRadii, const std::vector<int>& outerRadii,
  Dimensions2D scanDimensions, Dimensions2D frameDimensions,
  Coordinates2D center, int frameOffset);

} // namespace stempy

vector<STEMImage> createSTEMImages(const ElectronCountedDataPyArray& array,
                                   const vector<int>& innerRadii,
                                   const vector<int>& outerRadii,
                                   Coordinates2D coords)
{
  return createSTEMImages(array.data, innerRadii, outerRadii,
                          array.scanDimensions, array.frameDimensions, coords);
}

template <typename... Params>
ElectronCountedDataPyArray electronCountPy(Params&&... params)
{
  return electronCount(std::forward<Params>(params)...);
}

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

  py::class_<CalculateThresholdsResults<uint16_t>>(
    m, "_calculate_thresholds_results", py::buffer_protocol())
    .def_readonly("background_threshold",
                  &CalculateThresholdsResults<uint16_t>::backgroundThreshold)
    .def_readonly("xray_threshold",
                  &CalculateThresholdsResults<uint16_t>::xRayThreshold)
    .def_readonly("number_of_samples",
                  &CalculateThresholdsResults<uint16_t>::numberOfSamples)
    .def_readonly("min_sample",
                  &CalculateThresholdsResults<uint16_t>::minSample)
    .def_readonly("max_sample",
                  &CalculateThresholdsResults<uint16_t>::maxSample)
    .def_readonly("mean", &CalculateThresholdsResults<uint16_t>::mean)
    .def_readonly("variance", &CalculateThresholdsResults<uint16_t>::variance)
    .def_readonly("std_dev", &CalculateThresholdsResults<uint16_t>::stdDev)
    .def_readonly("number_of_bins",
                  &CalculateThresholdsResults<uint16_t>::numberOfBins)
    .def_readonly("xray_threshold_n_sigma",
                  &CalculateThresholdsResults<uint16_t>::xRayThresholdNSigma)
    .def_readonly(
      "background_threshold_n_sigma",
      &CalculateThresholdsResults<uint16_t>::backgroundThresholdNSigma)
    .def_readonly("optimized_mean",
                  &CalculateThresholdsResults<uint16_t>::optimizedMean)
    .def_readonly("optimized_std_dev",
                  &CalculateThresholdsResults<uint16_t>::optimizedStdDev);

  py::class_<CalculateThresholdsResults<float>>(
    m, "_calculate_thresholds_results_float", py::buffer_protocol())
    .def_readonly("background_threshold",
                  &CalculateThresholdsResults<float>::backgroundThreshold)
    .def_readonly("xray_threshold",
                  &CalculateThresholdsResults<float>::xRayThreshold)
    .def_readonly("number_of_samples",
                  &CalculateThresholdsResults<float>::numberOfSamples)
    .def_readonly("min_sample", &CalculateThresholdsResults<float>::minSample)
    .def_readonly("max_sample", &CalculateThresholdsResults<float>::maxSample)
    .def_readonly("mean", &CalculateThresholdsResults<float>::mean)
    .def_readonly("variance", &CalculateThresholdsResults<float>::variance)
    .def_readonly("std_dev", &CalculateThresholdsResults<float>::stdDev)
    .def_readonly("number_of_bins",
                  &CalculateThresholdsResults<float>::numberOfBins)
    .def_readonly("xray_threshold_n_sigma",
                  &CalculateThresholdsResults<float>::xRayThresholdNSigma)
    .def_readonly("background_threshold_n_sigma",
                  &CalculateThresholdsResults<float>::backgroundThresholdNSigma)
    .def_readonly("optimized_mean",
                  &CalculateThresholdsResults<float>::optimizedMean)
    .def_readonly("optimized_std_dev",
                  &CalculateThresholdsResults<float>::optimizedStdDev);

  py::class_<ElectronCountedData>(m, "_electron_counted_data",
                                  py::buffer_protocol())
    .def_readonly("data", &ElectronCountedData::data)
    .def_readonly("scan_positions", &ElectronCountedData::scanPositions)
    .def_readonly("scan_dimensions", &ElectronCountedData::scanDimensions)
    .def_readonly("frame_dimensions", &ElectronCountedData::frameDimensions);

  py::class_<ElectronCountedDataPyArray>(m, "_electron_counted_data_pyarray",
                                         py::buffer_protocol())
    .def_readonly("data", &ElectronCountedDataPyArray::data)
    .def_readonly("scan_positions", &ElectronCountedDataPyArray::scanPositions)
    .def_readonly("scan_dimensions",
                  &ElectronCountedDataPyArray::scanDimensions)
    .def_readonly("frame_dimensions",
                  &ElectronCountedDataPyArray::frameDimensions);

  // Add more template instantiations as we add more types of iterators
  m.def("create_stem_images",
        (vector<STEMImage>(*)(StreamReader::iterator, StreamReader::iterator,
                              const vector<int>&, const vector<int>&,
                              Dimensions2D, Coordinates2D)) &
          createSTEMImages<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "create_stem_images",
    (vector<STEMImage>(*)(SectorStreamReader::iterator,
                          SectorStreamReader::iterator, const vector<int>&,
                          const vector<int>&, Dimensions2D, Coordinates2D)) &
      createSTEMImages<SectorStreamReader::iterator>,
    py::call_guard<py::gil_scoped_release>());
  m.def("create_stem_images",
        (vector<STEMImage>(*)(const std::vector<py::array_t<uint32_t>>&,
                              const vector<int>&, const vector<int>&,
                              Dimensions2D, Dimensions2D, Coordinates2D, int)) &
          createSTEMImages,
        py::call_guard<py::gil_scoped_release>());
  m.def(
    "create_stem_images",
    (vector<STEMImage>(*)(const ElectronCountedDataPyArray&, const vector<int>&,
                          const vector<int>&, Coordinates2D)) &
      createSTEMImages,
    py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_average", &calculateAverage<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());

  // Electron counting without dark reference
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(StreamReader::iterator,
                                       StreamReader::iterator, double, double,
                                       py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator, double,
          double, py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(PyReader::iterator, PyReader::iterator,
                                       double, double, py::array_t<float>,
                                       Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());

  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamThreadedReader*, int, int,
                                       double, double, py::array_t<float>,
                                       Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamMultiPassThreadedReader*, int, int, double, double,
          py::array_t<float>, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());

  // Electron counting with gain and dark reference
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          StreamReader::iterator, StreamReader::iterator, Image<float>&, double,
          double, py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator,
          Image<float>&, double, double, py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(PyReader::iterator, PyReader::iterator,
                                       Image<float>&, double, double,
                                       py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          StreamReader::iterator, StreamReader::iterator, py::array_t<float>,
          double, double, py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamReader::iterator,
                                       SectorStreamReader::iterator,
                                       py::array_t<float>, double, double,
                                       py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(PyReader::iterator, PyReader::iterator,
                                       py::array_t<float>, double, double,
                                       py::array_t<float>, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamThreadedReader*, Image<float>&, int, int, double, double,
          py::array_t<float>, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamThreadedReader*, py::array_t<float>, int, int, double,
          double, py::array_t<float>, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamThreadedReader*, int, int, double, double,
                        Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamMultiPassThreadedReader*, Image<float>&, int, int, double,
          double, py::array_t<float>, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamMultiPassThreadedReader*, py::array_t<float>, int, int,
          double, double, py::array_t<float>, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamMultiPassThreadedReader*, int, int, double,
                        double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());

  // Electron counting, without gain
  m.def("electron_count",
        electronCountPy<StreamReader::iterator&, StreamReader::iterator&,
                        Image<float>&, double, double, Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamReader::iterator&,
                        SectorStreamReader::iterator&, Image<float>&, double,
                        double, Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<PyReader::iterator&, PyReader::iterator&, Image<float>&,
                        double, double, Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          StreamReader::iterator, StreamReader::iterator, py::array_t<float>,
          double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(
          SectorStreamReader::iterator, SectorStreamReader::iterator,
          py::array_t<float>, double, double, Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(PyReader::iterator, PyReader::iterator,
                                       py::array_t<float>, double, double,
                                       Dimensions2D)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamThreadedReader*, Image<float>&, int, int,
                        double, double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamThreadedReader*, int, int, double, double,
                        Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamThreadedReader*,
                                       py::array_t<float>, int, int, double,
                                       double, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamThreadedReader*, Image<float>&, int, int,
                        double, double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamThreadedReader*, int, int, double, double,
                        Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamThreadedReader*,
                                       py::array_t<float>, int, int, double,
                                       double, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());

  m.def("electron_count",
        electronCountPy<SectorStreamMultiPassThreadedReader*, Image<float>&,
                        int, int, double, double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamMultiPassThreadedReader*, int, int, double,
                        double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamMultiPassThreadedReader*,
                                       py::array_t<float>, int, int, double,
                                       double, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamMultiPassThreadedReader*, Image<float>&,
                        int, int, double, double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamMultiPassThreadedReader*, int, int, double,
                        double, Dimensions2D, bool>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        (ElectronCountedDataPyArray(*)(SectorStreamMultiPassThreadedReader*,
                                       py::array_t<float>, int, int, double,
                                       double, Dimensions2D, bool)) &
          electronCount,
        py::call_guard<py::gil_scoped_release>());

  // Electron counting without dark reference or gain
  m.def("electron_count",
        electronCountPy<StreamReader::iterator&, StreamReader::iterator&,
                        double, double, Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<SectorStreamReader::iterator&,
                        SectorStreamReader::iterator&, double, double,
                        Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());
  m.def("electron_count",
        electronCountPy<PyReader::iterator&, PyReader::iterator&, double,
                        double, Dimensions2D>,
        py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, with gain
  m.def(
    "calculate_thresholds",
    (CalculateThresholdsResults<float>(*)(vector<Block>&, Image<float>&, int,
                                          double, double, py::array_t<float>)) &
      calculateThresholds,
    py::call_guard<py::gil_scoped_release>());
  m.def(
    "calculate_thresholds",
    (CalculateThresholdsResults<float>(*)(vector<PyBlock>&, Image<float>&, int,
                                          double, double, py::array_t<float>)) &
      calculateThresholds,
    py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<Block>&,
                                              py::array_t<float>, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<PyBlock>&,
                                              py::array_t<float>, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, without gain
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<Block>&, Image<float>&,
                                                 int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(
          vector<PyBlock>&, Image<float>&, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(
          vector<Block>&, py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(
          vector<PyBlock>&, py::array_t<float>, int, double, double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, with gain and without darkreference
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<Block>&, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<PyBlock>&, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<Block>&, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<float>(*)(vector<PyBlock>&, int, double,
                                              double, py::array_t<float>)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());

  // Calculate thresholds, without gain and darkreference
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<Block>&, int, double,
                                                 double)) &
          calculateThresholds,
        py::call_guard<py::gil_scoped_release>());
  m.def("calculate_thresholds",
        (CalculateThresholdsResults<uint16_t>(*)(vector<PyBlock>&, int, double,
                                                 double)) &
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
  m.def("create_stem_images",
        (vector<STEMImage>(*)(PyReader::iterator, PyReader::iterator,
                              const vector<int>&, const vector<int>&,
                              Dimensions2D, Coordinates2D)) &
          createSTEMImages<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(StreamReader::iterator, StreamReader::iterator,
                          const Image<float>&)) &
          maximumDiffractionPattern<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(StreamReader::iterator, StreamReader::iterator)) &
          maximumDiffractionPattern<StreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(SectorStreamReader::iterator,
                          SectorStreamReader::iterator, const Image<float>&)) &
          maximumDiffractionPattern<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(SectorStreamReader::iterator,
                          SectorStreamReader::iterator)) &
          maximumDiffractionPattern<SectorStreamReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
  m.def("maximum_diffraction_pattern",
        (Image<double>(*)(PyReader::iterator, PyReader::iterator)) &
          maximumDiffractionPattern<PyReader::iterator>,
        py::call_guard<py::gil_scoped_release>());
}
