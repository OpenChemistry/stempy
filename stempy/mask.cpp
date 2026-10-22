#include "mask.h"

#include <cmath>

using std::pow;

namespace stempy {

uint16_t* createAnnularMask(Dimensions2D dimensions, int innerRadius,
                            int outerRadius, Coordinates2D center)
{
  return createAnnularMask(dimensions, innerRadius, outerRadius,
                                           CoordinatesDouble2D(center));
}

// Only typed double pairs select this overload. Bare brace-initialized centers
// cannot deduce Center and continue to use the original integer overload.
template <typename Center>
std::enable_if_t<std::is_same<Center, CoordinatesDouble2D>::value, uint16_t*>
createAnnularMask(Dimensions2D dimensions, int innerRadius,
                   int outerRadius, Center center)
{
  auto numberOfElements = dimensions.first * dimensions.second;
  auto mask = new uint16_t[numberOfElements]();

  if (center.first < 0)
    center.first = static_cast<int>(round(dimensions.first / 2.0));

  if (center.second < 0)
    center.second = static_cast<int>(round(dimensions.second / 2.0));

  innerRadius = static_cast<int>(pow(innerRadius, 2.0));
  outerRadius = static_cast<int>(pow(outerRadius, 2.0));

  for (uint32_t i = 0; i < numberOfElements; ++i) {
    // Ensure these are signed ints so we don't underflow below
    int x = i % dimensions.first;
    int y = i / dimensions.first;

    auto d = pow((x - center.first), 2.0) + pow((y - center.second), 2.0);
    mask[i] = d >= innerRadius && d < outerRadius ? 0xFFFF : 0;
  }

  return mask;
}

template uint16_t* createAnnularMask(Dimensions2D, int, int, CoordinatesDouble2D);

}
