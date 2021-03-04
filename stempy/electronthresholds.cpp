#include "electronthresholds.h"
#include "python/pyreader.h"

#include <algorithm>
#include <cmath>
#include <random>

#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>

namespace stempy {

template <typename T>
double calculateMean(std::vector<T>& values)
{
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

template <typename T>
double calculateVariance(std::vector<T>& values, double mean)
{
  double v1 = 0;
  double sigma2;

  for (size_t i = 0; i < values.size(); i++) {
    v1 += pow(values[i] - mean, 2.0);
  }
  sigma2 = v1 / (values.size() - 1.0);

  return sigma2;
}

struct GaussianErrorFunctor : Eigen::DenseFunctor<double>
{

  GaussianErrorFunctor(const Eigen::VectorXd& bins,
                       const Eigen::VectorXd& histogram)
    : Eigen::DenseFunctor<double>(3, bins.rows()), m_bins(bins),
      m_histogram(histogram)
  {}

  int operator()(const InputType& x, ValueType& fvec)
  {
    auto num = -(m_bins - ValueType::Constant(values(), x[1])).array().square();
    auto variance = pow(x[2], 2);

    fvec = x[0] * (num / (2 * variance)).exp() - m_histogram.array();

    return 0;
  }

  int df(const InputType& x, JacobianType& jacobian)
  {
    auto means = ValueType::Constant(values(), x[1]);
    auto tmp = (m_bins - means).array();
    auto num = -tmp.square();
    auto variance = pow(x[2], 2);
    auto den = 2 * variance;

    auto j0 = (num / den).exp();
    auto j1 = x[0] * tmp * j0 / variance;

    jacobian.col(0) = j0;
    jacobian.col(1) = j1;
    jacobian.col(2) = tmp * j1 / x[2];

    return 0;
  }

  Eigen::VectorXd m_bins;
  Eigen::VectorXd m_histogram;
};

template <typename BlockType, typename FrameType, bool dark>
CalculateThresholdsResults<FrameType> calculateThresholds(
  std::vector<BlockType>& blocks, const float darkReference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[])
{
  auto frameDimensions = blocks[0].header.frameDimensions;
  auto numberOfPixels = frameDimensions.first * frameDimensions.second;

  // Setup random number engine
  std::random_device randomDevice;
  std::default_random_engine randomEngine(randomDevice());

  int numberSamplePixels =
    frameDimensions.first * frameDimensions.second * numberOfSamples;
  std::vector<FrameType> samples(numberSamplePixels, 0);
  for (int i = 0; i < numberOfSamples; i++) {
    std::uniform_int_distribution<int> randomBlockDist(0, blocks.size() - 1);
    auto randomBlockIndex = randomBlockDist(randomEngine);
    auto randomBlock = blocks[randomBlockIndex];
    std::uniform_int_distribution<int> randomFrameDist(
      0, randomBlock.header.imagesInBlock - 1);
    auto randomFrameIndex = randomFrameDist(randomEngine);
    auto blockData = randomBlock.data.get();

    for (unsigned j = 0; j < numberOfPixels; j++) {
      // For now just use the index, the image number don't seem to work, in the
      // current data set. In the future we should be using the image number.

      // This will be evaluated a compile time.
      if (std::is_integral<FrameType>::value) {
        auto value = blockData[randomFrameIndex * numberOfPixels + j];
        static_if<dark>(
          [&]() { value -= static_cast<int16_t>(darkReference[j]); })();
        samples[i * numberOfPixels + j] = value;
      }
      // if not integral type then we know we have gain and need to multiple
      else {
        auto value = blockData[randomFrameIndex * numberOfPixels + j] * gain[j];
        static_if<dark>(
          [&]() { value -= static_cast<float>(darkReference[j]); })();
        samples[i * numberOfPixels + j] = value;
      }
    }
  }

  // Calculate stats
  auto mean = calculateMean(samples);
  auto variance = calculateVariance(samples, mean);
  auto stdDev = sqrt(variance);
  auto xrayThreshold = mean + xRayThresholdNSigma * stdDev;

  // Now generate a histograms
  auto minMax = std::minmax_element(samples.begin(), samples.end());
  auto minSample = *minMax.first;
  auto maxSample = static_cast<FrameType>(std::ceil(*minMax.second));
  auto maxBin = std::min(static_cast<int>(maxSample),
                         static_cast<int>(mean + xrayThreshold * stdDev));
  auto minBin = std::max(static_cast<int>(minSample),
                         static_cast<int>(mean - xrayThreshold * stdDev));

  auto numberOfBins = maxBin - minBin;
  std::vector<double> histogram(numberOfBins, 0.0);
  std::vector<double> bins(numberOfBins);

  auto binEdge = minBin;
  for (int i = 0; i < numberOfBins; i++) {
    bins[i] = binEdge++;
  }

  // Bin the values
  for (int i = 0; i < numberSamplePixels; i++) {
    auto binIndex = static_cast<int>(samples[i] - minBin);
    // Skip values outside range
    if (binIndex >= numberOfBins) {
      continue;
    }
    histogram[binIndex] += 1;
  }

  auto b = Eigen::Map<Eigen::VectorXd>(bins.data(), bins.size());
  auto h = Eigen::Map<Eigen::VectorXd>(histogram.data(), histogram.size());

  GaussianErrorFunctor gef(b, h);
  Eigen::VectorXd state(3);
  auto indexOfMaxElement =
    std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
  state << static_cast<double>(histogram[indexOfMaxElement]), mean, stdDev;

  Eigen::LevenbergMarquardt<GaussianErrorFunctor> solver(gef);
  solver.minimize(state);

  if (solver.info() != Eigen::ComputationInfo::Success) {
    throw std::runtime_error("Optimization did not converge");
  }

  auto optimizedMean = state[1];
  auto optimizedStdDev = fabs(state[2]);

  auto backgroundThreshold =
    optimizedMean + optimizedStdDev * backgroundThresholdNSigma;

  CalculateThresholdsResults<FrameType> ret;
  ret.numberOfSamples = numberOfSamples;
  ret.minSample = minSample;
  ret.maxSample = maxSample;
  ret.mean = mean;
  ret.variance = variance;
  ret.stdDev = stdDev;
  ret.numberOfBins = numberOfBins;
  ret.xRayThresholdNSigma = xRayThresholdNSigma;
  ret.backgroundThresholdNSigma = backgroundThresholdNSigma;
  ret.xRayThreshold = xrayThreshold;
  ret.backgroundThreshold = backgroundThreshold;
  ret.optimizedMean = optimizedMean;
  ret.optimizedStdDev = optimizedStdDev;

  return ret;
}

// Without gain
template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, Image<float>& darkreference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds<BlockType, uint16_t>(
    blocks, darkreference.data.get(), numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, nullptr);
}

template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, const float darkreference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma)
{
  return calculateThresholds<BlockType, uint16_t>(
    blocks, darkreference, numberOfSamples, backgroundThresholdNSigma,
    xRayThresholdNSigma, nullptr);
}

// With gain
template <typename BlockType>
CalculateThresholdsResults<float> calculateThresholds(
  std::vector<BlockType>& blocks, Image<float>& darkreference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[])
{
  return calculateThresholds<BlockType, float>(
    blocks, darkreference.data.get(), numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, gain);
}

// Without gain, without darkreference
template <typename BlockType>
CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<BlockType>& blocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma)
{
  return calculateThresholds<BlockType, uint16_t, false>(
    blocks, nullptr, numberOfSamples, backgroundThresholdNSigma,
    xRayThresholdNSigma, nullptr);
}

// With gain
template CalculateThresholdsResults<float> calculateThresholds<Block>(
  std::vector<Block>& blocks, Image<float>& darkReference, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[]);
template CalculateThresholdsResults<float> calculateThresholds<PyBlock>(
  std::vector<PyBlock>& blocks, Image<float>& darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[]);
template CalculateThresholdsResults<float> calculateThresholds<Block, float, true>(
  std::vector<Block>& blocks, const float darkReference[], int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[]);
template CalculateThresholdsResults<uint16_t> calculateThresholds<Block>(
  std::vector<Block>& blocks, const float darkReference[], int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[]);
template CalculateThresholdsResults<float> calculateThresholds<PyBlock, float, true>(
  std::vector<PyBlock>& blocks, const float darkReference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[]);

// No gain
template CalculateThresholdsResults<uint16_t> calculateThresholds<Block>(
  std::vector<Block>& blocks, Image<float>& darkReference, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma);
template CalculateThresholdsResults<uint16_t> calculateThresholds<PyBlock>(
  std::vector<PyBlock>& blocks, Image<float>& darkReference,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);
template CalculateThresholdsResults<uint16_t> calculateThresholds<Block>(
  std::vector<Block>& blocks, const float darkReference[], int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma);
template CalculateThresholdsResults<uint16_t> calculateThresholds<PyBlock>(
  std::vector<PyBlock>& blocks, const float darkReference[],
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma);

template CalculateThresholdsResults<uint16_t>
calculateThresholds<Block, uint16_t, false>(std::vector<Block>& blocks,
                                            const float darkReference[],
                                            int numberOfSamples,
                                            double backgroundThresholdNSigma,
                                            double xRayThresholdNSigma,
                                            const float gain[]);
template CalculateThresholdsResults<uint16_t>
calculateThresholds<PyBlock, uint16_t, false>(std::vector<PyBlock>& blocks,
                                              const float darkReference[],
                                              int numberOfSamples,
                                              double backgroundThresholdNSigma,
                                              double xRayThresholdNSigma,
                                              const float gain[]);

template CalculateThresholdsResults<float>
calculateThresholds<Block, float, false>(std::vector<Block>& blocks,
                                         const float darkReference[],
                                         int numberOfSamples,
                                         double backgroundThresholdNSigma,
                                         double xRayThresholdNSigma,
                                         const float gain[]);
template CalculateThresholdsResults<float>
calculateThresholds<PyBlock, float, false>(std::vector<PyBlock>& blocks,
                                           const float darkReference[],
                                           int numberOfSamples,
                                           double backgroundThresholdNSigma,
                                           double xRayThresholdNSigma,
                                           const float gain[]);

// No gain or dark reference
template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<Block>& blocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma);

template CalculateThresholdsResults<uint16_t> calculateThresholds(
  std::vector<PyBlock>& blocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma);
}
