#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include "image.h"

namespace stempy {

struct ElectronCountedData
{
  std::vector<std::vector<uint32_t>> data;

  // These types match those of the Header class
  uint16_t scanWidth = 0;
  uint16_t scanHeight = 0;
  uint32_t frameWidth = 0;
  uint32_t frameHeight = 0;
};

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<double>& darkreference,
                                  double backgroundThreshold,
                                  double xRayThreshold, int scanWidth = 0,
                                  int scanHeight = 0);

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const double darkreference[],
                                  double backgroundThreshold,
                                  double xRayThreshold, int scanWidth = 0,
                                  int scanHeight = 0);
}

#endif /* STEMPY_ELECTRON_H_ */
