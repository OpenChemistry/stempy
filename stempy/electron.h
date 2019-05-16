#ifndef STEMPY_ELECTRON_H_
#define STEMPY_ELECTRON_H_

#include "image.h"

namespace stempy {

template <typename InputIt>
std::vector<std::vector<uint32_t>> electronCount(InputIt first, InputIt last,
                                                 Image<double>& darkreference,
                                                 double backgroundThreshold,
                                                 double xRayThreshold,
                                                 int scanWidth = 0,
                                                 int scanHeight = 0);
}

#endif /* STEMPY_ELECTRON_H_ */
