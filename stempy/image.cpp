#include "image.h"

#include "config.h"
#include "mask.h"

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#endif

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

STEMValues calculateSTEMValues(uint16_t data[], int offset,
                               int numberOfPixels,
                               uint16_t brightFieldMask[],
                               uint16_t darkFieldMask[],
                               uint32_t imageNumber)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;
  for (int i=0; i<numberOfPixels; i++) {
    auto value = data[offset + i];

    stemValues.bright += value & brightFieldMask[i];
    stemValues.dark  += value & darkFieldMask[i];
  }

  return stemValues;
}

#ifdef VTKm
namespace {
  struct MaskAndAdd
  {
    VTKM_EXEC_CONT
    uint64_t operator()(const vtkm::Pair<uint16_t,uint16_t>& a, const vtkm::Pair<uint16_t,uint16_t>& b) const
    {
      return (a.first & a.second) + (b.first & b.second);
    }

    VTKM_EXEC_CONT
    uint64_t operator()(uint64_t a, uint64_t b) const
    {
      return a + b;
    }

    VTKM_EXEC_CONT
    uint64_t operator()(const vtkm::Pair<uint16_t,uint16_t>& a, uint64_t b) const
    {
      return (a.first & a.second) + b;
    }

    VTKM_EXEC_CONT
    uint64_t operator()(uint64_t a, const vtkm::Pair<uint16_t,uint16_t>& b) const
    {
      return a + (b.first & b.second);
    }
  };
}

STEMValues calculateSTEMValuesParallel(uint16_t data[], int offset,
                                       int numberOfPixels,
                                       uint16_t brightFieldMask[],
                                       uint16_t darkFieldMask[],
                                       uint32_t imageNumber)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;

  auto keysBright = vtkm::cont::make_ArrayHandle(brightFieldMask,
                                                 numberOfPixels);
  auto keysDark = vtkm::cont::make_ArrayHandle(darkFieldMask, numberOfPixels);
  auto input = vtkm::cont::make_ArrayHandle(&data[offset], numberOfPixels);

  auto inputAndBright = vtkm::cont::make_ArrayHandleZip(input, keysBright);
  auto inputAndDark = vtkm::cont::make_ArrayHandleZip(input, keysDark);

  const uint64_t initialVal = 0;
  stemValues.bright = vtkm::cont::Algorithm::Reduce(inputAndBright, initialVal,
                                                    MaskAndAdd{} );
  stemValues.dark = vtkm::cont::Algorithm::Reduce(inputAndDark, initialVal,
                                                  MaskAndAdd{} );

  return stemValues;
}
#endif

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
      auto stemValues = calculateSTEMValues(data, i*numberOfPixels, numberOfPixels,
                                            brightFieldMask, darkFieldMask);
      image.bright.data[block.header.imageNumbers[i]-1] = stemValues.bright;
      image.dark.data[block.header.imageNumbers[i]-1] = stemValues.dark;
    }
  }

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}
}
