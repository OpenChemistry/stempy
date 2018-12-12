#include "image.h"
#include "mask.h"

#include <memory>

using namespace std;

namespace stempy {

Image::Image(uint32_t width, uint32_t height) :
    width(width), height(height),
    data(new uint64_t[width * height], std::default_delete<uint64_t[]>())
{ }

STEMImage::STEMImage(uint32_t width, uint32_t height)
{
  this->bright = Image(width, height);
  this->dark = Image(width, height);
}

STEMImage createSTEMImage(std::vector<Block>& blocks, int rows, int columns,  int innerRadius, int outerRadius)
{
  STEMImage image(rows, columns);

  // Get image size from first block
  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  auto brightFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, 0, outerRadius);
  auto darkFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, innerRadius, outerRadius);

  for(const Block &block: blocks) {
    auto data = block.data.get();
    for (int i=0; i<block.header.imagesInBlock; i++) {
      uint64_t bright=0, dark=0;
      for (int j=0; j<numberOfPixels; j++) {
        auto value = data[i * numberOfPixels + j];

        bright += value & brightFieldMask[j];
        dark  += value & darkFieldMask[j];
      }

      image.bright.data[block.header.imageNumbers[i]-1] = bright;
      image.dark.data[block.header.imageNumbers[i]-1] = dark;
    }
  }

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}
}
