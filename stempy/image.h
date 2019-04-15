#ifndef stempyimages_h
#define stempyimages_h

#include "reader.h"

#include <vector>
#include <memory>

namespace stempy {

  template <typename T>
  struct Image {
    std::shared_ptr<T[]> data;
    uint32_t width = 0;
    uint32_t height = 0;

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
  STEMImage createSTEMImage(InputIt first, InputIt last, int rows, int colums,
                            int innerRadius, int outerRadius);
  STEMValues calculateSTEMValues(uint16_t data[], int offset,
                                 int numberOfPixels,
                                 uint16_t brightFieldMask[],
                                 uint16_t darkFieldMask[],
                                 uint32_t imageNumber=-1);

  Image<double> calculateAverage(std::vector<Block> &blocks);
}

#endif
