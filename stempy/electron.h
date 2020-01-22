#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include "image.h"

namespace stempy {

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
}

#endif /* STEMPY_ELECTRON_H_ */
