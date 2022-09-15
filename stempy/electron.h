#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include <cfloat>

#include "image.h"

namespace stempy {

class SectorStreamThreadedReader;

using Events = std::vector<std::vector<std::vector<uint32_t>>>;

struct ElectronCountedMetadata
{
  bool thresholdCalculated = true;
  double backgroundThreshold = 0.0;
  double xRayThreshold = 0.0;
  int numberOfSamples = 0;
  double minSample = 0;
  double maxSample = 0;
  double mean = 0.0;
  double variance = 0.0;
  double stdDev = 0.0;
  int numberOfBins = 0;
  double xRayThresholdNSigma = 0.0;
  double backgroundThresholdNSigma = 0.0;
  double optimizedMean = 0.0;
  double optimizedStdDev = 0.0;
};

struct ElectronCountedData
{
  Events data;

  ElectronCountedMetadata metadata;
  Dimensions2D scanDimensions = { 0, 0 };
  Dimensions2D frameDimensions = { 0, 0 };
};

struct ElectronCountOptionsClassic
{
  float* darkReference = nullptr;
  double backgroundThreshold = DBL_MIN;
  double xRayThreshold = DBL_MAX;
  float* gain = nullptr;
  Dimensions2D scanDimensions = { 0, 0 };
  bool applyRowDarkSubtraction = false;
  float optimizedMean = 0;
  bool applyRowDarkUseMean = true;
};

struct ElectronCountOptions
{
  float* darkReference = nullptr;
  int thresholdNumberOfBlocks = 1;
  int numberOfSamples = 20;
  double backgroundThresholdNSigma = 4;
  double xRayThresholdNSigma = 10;
  float* gain = nullptr;
  Dimensions2D scanDimensions = { 0, 0 };
  bool verbose = false;
  bool applyRowDarkSubtraction = false;
  bool applyRowDarkUseMean = true;
};

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const ElectronCountOptionsClassic& options);

template <typename Reader>
ElectronCountedData electronCount(Reader* reader,
                                  const ElectronCountOptions& options);

#ifdef USE_MPI

void initMpiWorldRank(int& worldSize, int& rank);
int getSampleBlocksPerRank(int worldSize, int rank,
                           int thresholdNumberOfBlocks);
void gatherBlocks(int worldSize, int rank, std::vector<Block>& blocks);
void gatherEvents(int worldSize, int rank, Events& events);
void broadcastThresholds(double& background, double& xRay,
                         double& optimizedMean);

#endif
}

#endif /* STEMPY_ELECTRON_H_ */
