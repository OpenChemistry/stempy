#include "mask.h"

#include <cmath>


using namespace std;

namespace stempy {

uint16_t* createAnnularMask(int rows, int columns, int innerRadius, int outerRadius) {
  auto numberOfElements = rows*columns;
  auto mask = new uint16_t[numberOfElements]();
  auto xCenter = round(rows/2.0);
  auto yCenter = round(rows/2.0);
  innerRadius = pow(innerRadius, 2.0);
  outerRadius = pow(outerRadius, 2.0);

  for (int i=0; i<numberOfElements; i++) {
    auto x = i % rows;
    auto y = i / rows;

    auto d = pow((x-xCenter), 2.0) + pow((y-yCenter), 2.0);
    mask[i] = d >= innerRadius && d < outerRadius ? 0xFFFF : 0;
  }

  return mask;
}

}
