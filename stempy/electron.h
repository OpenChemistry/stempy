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

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<double>& darkreference,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const double darkreference[],
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<double>& darkreference,
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions = { 0, 0 });

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const double darkreference[],
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

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, Image<double>& darkreference,
  int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, const double darkreference[],
  int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, Image<double>& darkreference,
  int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  const float gain[] = nullptr, Dimensions2D scanDimensions = { 0, 0 },
  bool verbose = false);

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, const double darkreference[],
  int thresholdNumberOfBlocks = 1, int numberOfSamples = 20,
  double backgroundThresholdNSigma = 4, double xRayThresholdNSigma = 10,
  const float gain[] = nullptr, Dimensions2D scanDimensions = { 0, 0 },
  bool verbose = false);

ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, int thresholdNumberOfBlocks = 1,
  int numberOfSamples = 20, double backgroundThresholdNSigma = 4,
  double xRayThresholdNSigma = 10, const float gain[] = nullptr,
  Dimensions2D scanDimensions = { 0, 0 }, bool verbose = false);

ElectronCountedData electronCount(SectorStreamThreadedReader* reader,
                                  int thresholdNumberOfBlocks = 1,
                                  int numberOfSamples = 20,
                                  double backgroundThresholdNSigma = 4,
                                  double xRayThresholdNSigma = 10,
                                  Dimensions2D scanDimensions = { 0, 0 },
                                  bool verbose = false);
}

#endif /* STEMPY_ELECTRON_H_ */
