#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include "image.h"

namespace stempy {

class SectorStreamThreadedReader;

struct ElectronCountedData
{
  std::vector<std::vector<uint32_t>> data;

  Dimensions2D scanDimensions = { 0, 0 };
  Dimensions2D frameDimensions = { 0, 0 };
};

using Events = std::vector<std::vector<uint32_t>>;

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<float>& darkreference,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const float darkreference[],
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<float>& darkreference,
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const float darkreference[],
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, Image<float>& darkreference,
                                  int thresholdNumberOfBlocks = 1,
                                  int numberOfSamples = 20,
                                  double backgroundThresholdNSigma = 4,
                                  double xRayThresholdNSigma = 10,
                                  Dimensions2D scanDimensions = { 0, 0 },
                                  bool verbose = false);

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, const float darkreference[],
                                  int thresholdNumberOfBlocks = 1,
                                  int numberOfSamples = 20,
                                  double backgroundThresholdNSigma = 4,
                                  double xRayThresholdNSigma = 10,
                                  Dimensions2D scanDimensions = { 0, 0 },
                                  bool verbose = false);

template <typename Reader>
ElectronCountedData electronCount(
  Reader* reader, Image<float>& darkreference, int thresholdNumberOfBlocks = 1,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10, const float gain[] = nullptr,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

template <typename Reader>
ElectronCountedData electronCount(
  Reader* reader, const float darkreference[], int thresholdNumberOfBlocks = 1,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10, const float gain[] = nullptr,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

template <typename Reader>
ElectronCountedData electronCount(
  Reader* reader, int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  const float gain[] = nullptr, Dimensions2D scanDimensions = { 0, 0 },
  bool verbose = false);

template <typename Reader>
ElectronCountedData electronCount(
  Reader* reader, int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

#ifdef USE_MPI

void initMpiWorldRank(int& worldSize, int& rank);
int getSampleBlocksPerRank(int worldSize, int rank,
                           int thresholdNumberOfBlocks);
void gatherBlocks(int worldSize, int rank, std::vector<Block>& blocks);
void gatherEvents(int worldSize, int rank, Events& events);
void broadcastThresholds(double& background, double& xRay);

#endif
}

#endif /* STEMPY_ELECTRON_H_ */
