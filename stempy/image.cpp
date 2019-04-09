#include "image.h"

#include "config.h"
#include "mask.h"

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#endif

#include <memory>

using namespace std;

namespace stempy {

template<typename T>
Image<T>::Image(uint32_t width, uint32_t height) :
    width(width), height(height),
    data(new T[width * height], std::default_delete<T[]>())
{ }

STEMImage::STEMImage(uint32_t width, uint32_t height)
{
  this->bright = Image<uint64_t>(width, height);
  this->dark = Image<uint64_t>(width, height);
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
                                       uint32_t imageNumber = -1)
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
    for(int i=0; i<block.header.imagesInBlock; i++) {
#ifdef VTKm
      auto stemValues = calculateSTEMValuesParallel(data, i*numberOfPixels, numberOfPixels,
                                                    brightFieldMask, darkFieldMask);
#else
      auto stemValues = calculateSTEMValues(data, i*numberOfPixels, numberOfPixels,
                                            brightFieldMask, darkFieldMask);
#endif
      image.bright.data[block.header.imageNumbers[i]-1] = stemValues.bright;
      image.dark.data[block.header.imageNumbers[i]-1] = stemValues.dark;
    }
  }

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}

Image<double> calculateAverage(std::vector<Block> &blocks)
{
  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows*detectorImageColumns;
  Image<double> image(detectorImageRows, detectorImageColumns);

  std::fill(image.data.get(), image.data.get() + numberOfPixels, 0.0);
  uint64_t numberOfImages = 0;
  for(const Block &block: blocks) {
    auto blockData = block.data.get();
    numberOfImages += block.header.imagesInBlock;
    for(int i=0; i<block.header.imagesInBlock; i++) {
      auto numberOfPixels = block.header.rows * block.header.columns;
      for(int j=0; j<numberOfPixels; j++) {
        image.data[j] += blockData[i*numberOfPixels+j];
      }
    }
  }

  for(int i=0; i<detectorImageRows*detectorImageColumns; i++) {
    image.data[i] /= numberOfImages;
  }

  return image;
}


}
