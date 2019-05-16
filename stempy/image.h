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
    uint64_t bright = 0;
    uint64_t dark = 0;
    uint32_t imageNumber = -1;
  };

  struct STEMImage {
    Image<uint64_t> bright;
    Image<uint64_t> dark;

    STEMImage() = default;
    STEMImage(uint32_t width, uint32_t height);
    STEMImage(STEMImage&& i) noexcept = default;
    STEMImage& operator=(STEMImage&& i) noexcept = default;
  };

  template <typename InputIt>
  STEMImage createSTEMImage(InputIt first, InputIt last, int innerRadius,
                            int outerRadius, int scanWidth = 0, int scanHeight = 0,
                            int centerX = -1, int centerY = -1);

  STEMImage createSTEMImageSparse(
    const std::vector<std::vector<uint32_t>>& sparseData, int innerRadius,
    int outerRadius, int rows, int columns, int frameWidth, int frameHeight,
    int centerX = -1, int centerY = -1);

  STEMValues calculateSTEMValues(const uint16_t data[], int offset,
                                 int numberOfPixels,
                                 uint16_t brightFieldMask[],
                                 uint16_t darkFieldMask[],
                                 uint32_t imageNumber = -1);

  template <typename InputIt>
  Image<double> calculateAverage(InputIt first, InputIt last);
}

#endif
