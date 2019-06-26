#include "mask.h"

#include <cmath>

using std::pow;

namespace stempy {

uint16_t* createAnnularMask(int width, int height, int innerRadius,
                            int outerRadius, int centerX, int centerY)
{
  auto numberOfElements = width*height;
  auto mask = new uint16_t[numberOfElements]();

  if (centerX < 0)
    centerX = round(width / 2.0);

  if (centerY < 0)
    centerY = round(height / 2.0);

  innerRadius = static_cast<int>(pow(innerRadius, 2.0));
  outerRadius = static_cast<int>(pow(outerRadius, 2.0));

  for (int i=0; i<numberOfElements; i++) {
    auto x = i % width;
    auto y = i / width;

    auto d = pow((x - centerX), 2.0) + pow((y - centerY), 2.0);
    mask[i] = d >= innerRadius && d < outerRadius ? 0xFFFF : 0;
  }

  return mask;
}

}
