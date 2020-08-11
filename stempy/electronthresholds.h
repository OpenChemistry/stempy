#ifndef STEMPY_ELECTRONTHRESHOLDS_H_
#define STEMPY_ELECTRONTHRESHOLDS_H_

#include "image.h"
#include "python/pyreader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace stempy {

template <typename FrameType>
struct CalculateThresholdsResults
{
  double backgroundThreshold = 0.0;
  double xRayThreshold = 0.0;
  int numberOfSamples = 0;
  FrameType minSample = 0;
  FrameType maxSample = 0;
  double mean = 0.0;
  double variance = 0.0;
  double stdDev = 0.0;
  int numberOfBins = 0;
  double xRayThresholdNSigma = 0.0;
  double backgroundThresholdNSigma = 0.0;
  double optimizedMean = 0.0;
  double optimizedStdDev = 0.0;
};

template <typename BlockType, typename FrameType, bool dark = true>
CalculateThresholdsResults<FrameType> calculateThresholds(
  std::vector<BlockType>& blocks, const double darkreference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[]);

// Without gain
template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, Image<double>& darkreference,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10);

template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, const double darkreference[],
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10);

template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<double> darkreference,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10);

// With gain
template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, Image<double>& darkreference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[]);

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, const double darkreference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[]);

template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, py::array_t<double> darkreference,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10);

// Without gain, without darkreference
template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10);
}

template <bool>
struct tag
{};

template <typename T, typename F>
auto static_if(tag<true>, T t, F f)
{
  (void)f;
  return t;
}

template <typename T, typename F>
auto static_if(tag<false>, T t, F f)
{
  (void)t;
  return f;
}

template <bool B, typename T, typename F>
auto static_if(T t, F f)
{
  return static_if(tag<B>{}, t, f);
}

template <bool B, typename T>
auto static_if(T t)
{
  return static_if(tag<B>{}, t, [](auto&&...) {});
}

#endif /* STEMPY_ELECTRONTHRESHOLDS_H_ */
