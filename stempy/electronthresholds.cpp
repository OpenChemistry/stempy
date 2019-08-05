#include "electronthresholds.h"

#include <algorithm>
#include <cmath>
#include <random>

#include <lsq/lsqcpp.h>

namespace stempy {

double calculateMean(std::vector<int16_t>& values)
{
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculateVariance(std::vector<int16_t>& values, double mean)
{
  double variance = 0;

  for (size_t i = 0; i < values.size(); i++) {
    variance += pow(values[i], 2.0);
  }

  return (variance / (values.size() - 1)) - pow(mean, 2.0);
}

class GaussianErrorFunction : public lsq::ErrorFunction<double>
{

public:
  GaussianErrorFunction(const std::vector<int32_t>& b,
                        const std::vector<uint64_t>& h)
    : bins(b), histogram(h)
  {
  }
  void _evaluate(const lsq::Vectord& state, lsq::Vectord& outVal,
                 lsq::Matrixd&) override
  {
    outVal.resize(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) {
      outVal[i] =
        (state[0] * exp(-0.5 * pow((this->bins[i] - state[1]) / state[2], 2)) -
         histogram[i]);
    }
  }

private:
  const std::vector<int32_t>& bins;
  const std::vector<uint64_t>& histogram;
};

CalculateThresholdsResults calculateThresholds(std::vector<Block>& blocks,
                                               Image<double>& darkReference,
                                               int numberOfSamples,
                                               double backgroundThresholdNSigma,
                                               double xRayThresholdNSigma)
{
  auto frameWidth = blocks[0].header.frameWidth;
  auto frameHeight = blocks[0].header.frameHeight;
  auto numberOfPixels = frameWidth * frameHeight;

  // Setup random number engine
  std::random_device randomDevice;
  std::default_random_engine randomEngine(randomDevice());

  int numberSamplePixels =
    frameWidth * frameHeight * numberOfSamples;
  std::vector<int16_t> samples(numberSamplePixels, 0);
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
      samples[i * numberOfPixels + j] =
        blockData[randomFrameIndex * numberOfPixels + j] -
        static_cast<int16_t>(darkReference.data[j]);
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
  auto maxSample = std::ceil(*minMax.second);
  auto maxBin = std::min(static_cast<int>(maxSample),
                         static_cast<int>(mean + xrayThreshold * stdDev));
  auto minBin = std::max(static_cast<int>(minSample),
                         static_cast<int>(mean - xrayThreshold * stdDev));

  auto numberOfBins = maxBin - minBin;
  std::vector<uint64_t> histogram(numberOfBins, 0);
  std::vector<int32_t> bins(numberOfBins);

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

  // Now optimize to file Gaussian
  lsq::LevenbergMarquardt<double> optAlgo;
  optAlgo.setLineSearchAlgorithm(nullptr);
  optAlgo.setMaxIterationsLM(20);
  optAlgo.setEpsilon(1e-6);
  GaussianErrorFunction* errorFunction =
    new GaussianErrorFunction(bins, histogram);
  optAlgo.setErrorFunction(errorFunction);
  lsq::Vectord initialState(3);
  auto indexOfMaxElement =
    std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
  initialState[0] = static_cast<double>(histogram[indexOfMaxElement]);
  initialState[1] =
    (bins[indexOfMaxElement + 1] - bins[indexOfMaxElement]) / 2.0;
  initialState[2] = stdDev;

  // optimize
  auto result = optAlgo.optimize(initialState);

  if (!result.converged) {
    throw std::runtime_error("Optimization did not converge");
  }
  auto backgroundThreshold =
    result.state[1] + result.state[2] * backgroundThresholdNSigma;

  CalculateThresholdsResults ret;
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

  return ret;
}

}
