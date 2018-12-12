#ifndef stempyimages_h
#define stempyimages_h

#include "reader.h"

#include <vector>
#include <memory>


namespace stempy {

  struct Image {
    std::shared_ptr<uint64_t[]> data;
    uint32_t width = 0;
    uint32_t height = 0;

    Image() = default;
    Image(uint32_t width, uint32_t height);
    Image(Image&& i) noexcept = default;
    Image& operator=(Image&& i) noexcept = default;
  };

  struct STEMImage {
    Image bright;
    Image dark;

    STEMImage() = default;
    STEMImage(uint32_t width, uint32_t height);
    STEMImage(STEMImage&& i) noexcept = default;
    STEMImage& operator=(STEMImage&& i) noexcept = default;
  };

  STEMImage createSTEMImage(std::vector<Block> &blocks, int rows, int colums,  int innerRadius, int outerRadius);

}

#endif
