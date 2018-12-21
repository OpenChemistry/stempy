#ifndef stempymask_h
#define stempymask_h

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace stempy {

  uint16_t* createAnnularMask(int rows, int columns, int innerRadius, int outerRadius);

}

#endif
