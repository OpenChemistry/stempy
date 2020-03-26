#ifndef stempymask_h
#define stempymask_h

#include "reader.h"

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace stempy {

uint16_t* createAnnularMask(Dimensions2D dimensions, int innerRadius,
                            int outerRadius, Coordinates2D center = { -1, -1 });
}

#endif
