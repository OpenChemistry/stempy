#ifndef STEMPY_ELECTRONTHRESHOLDS_H_
#define STEMPY_ELECTRONTHRESHOLDS_H_

#include "image.h"

namespace stempy {

struct CalculateThresholdsResults
{
  double backgroundThreshold = 0.0;
  double xRayThreshold = 0.0;
  int numberOfSamples = 0;
  int16_t minSample = 0;
  int16_t maxSample = 0;
  double mean = 0.0;
  double variance = 0.0;
  double stdDev = 0.0;
  int numberOfBins = 0;
  double xRayThresholdNSigma = 0.0;
  double backgroundThresholdNSigma = 0.0;
};

CalculateThresholdsResults calculateThresholds(
  std::vector<Block>& blocks, Image<double>& darkreference,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10);
}

#endif /* STEMPY_ELECTRONTHRESHOLDS_H_ */
