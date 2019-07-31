#ifndef STEMPY_ELECTRONTHRESHOLDS_H_
#define STEMPY_ELECTRONTHRESHOLDS_H_

#include "image.h"

namespace stempy {

std::pair<double, double> calculateThresholds(std::vector<Block>& blocks,
                                              Image<double>& darkreference,
                                              int numberOfSamples = 20,
                                              double backgroundThresholdNSigma = 4,
                                              double xRayThresholdNSigma = 10);
}

#endif /* STEMPY_ELECTRONTHRESHOLDS_H_ */
