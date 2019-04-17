#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include "image.h"

namespace stempy {

std::pair<double, double> calculateThresholds(std::vector<Block>& blocks,
                                              Image<double>& darkreference,
                                              int numberOfSamples = 20,
                                              int backgroundThresholdNSigma = 4,
                                              int xRayThresholdNSigma = 10);

std::vector<std::vector<uint32_t>> electronCount(
  std::vector<Block>& blocks, int scanRows, int scanColumns,
  Image<double>& darkreference, int numberOfSamples = 20,
  int backgroundThresholdNSigma = 4, int xRayThresholdNSigma = 30);
}

#endif /* STEMPY_ELECTRON_H_ */
