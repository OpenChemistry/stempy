#ifndef stempymask_h
#define stempymask_h

#include "reader.h"

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <type_traits>

namespace stempy {

// Preserve the integer API, including brace-initialized centers.
uint16_t* createAnnularMask(Dimensions2D dimensions, int innerRadius,
                            int outerRadius, Coordinates2D center = { -1, -1 });

// Only typed double pairs select this overload. Bare brace-initialized centers
// cannot deduce Center and continue to use the original integer overload.
template <typename Center>
std::enable_if_t<std::is_same<Center, CoordinatesDouble2D>::value, uint16_t*>
createAnnularMask(Dimensions2D dimensions, int innerRadius,
                   int outerRadius, Center center);
}

#endif
