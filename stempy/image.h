#ifndef stempyimages_h
#define stempyimages_h

#include "reader.h"

#include <vector>
#include <memory>

namespace stempy {

  template <typename T>
  struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::shared_ptr<T[]> data;

    Image() = default;
    Image(uint32_t width, uint32_t height);
    Image(Image&& i) noexcept = default;
    Image& operator=(Image&& i) noexcept = default;
  };

  struct STEMValues {
    uint64_t data = 0;
    uint32_t imageNumber = -1;
  };

  template <typename T>
  struct RadialSum {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t radii = 0;
    std::shared_ptr<T[]> data;

    RadialSum() = default;
    RadialSum(uint32_t width, uint32_t height, uint32_t radii);
    RadialSum(RadialSum&& i) noexcept = default;
    RadialSum& operator=(RadialSum&& i) noexcept = default;
  };

  using STEMImage = Image<uint64_t>;

  template <typename InputIt>
  std::vector<STEMImage> createSTEMImages(InputIt first, InputIt last,
                                          std::vector<int> innerRadii,
                                          std::vector<int> outerRadii,
                                          int scanWidth = 0, int scanHeight = 0,
                                          int centerX = -1, int centerY = -1);

  std::vector<STEMImage> createSTEMImagesSparse(
    const std::vector<std::vector<uint32_t>>& sparseData,
    std::vector<int> innerRadii, std::vector<int> outerRadii, int rows,
    int columns, int frameWidth, int frameHeight, int centerX = -1,
    int centerY = -1);

  STEMValues calculateSTEMValues(const uint16_t data[], int offset,
                                 int numberOfPixels, uint16_t mask[],
                                 uint32_t imageNumber = -1);

  template <typename InputIt>
  Image<double> calculateAverage(InputIt first, InputIt last);


  template <typename InputIt>
  RadialSum<uint64_t> radialSum(InputIt first, InputIt last, int scanWidth = 0, int scanHeight = 0,
        int centerX = -1, int centerY = -1);
}

#endif
