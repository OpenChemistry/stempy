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
    using MasksType = vtkm::Pair<uint16_t,uint16_t>;
    using MasksAndInputType = vtkm::Pair<MasksType, uint16_t>;
    using OutputType = vtkm::Pair<uint64_t, uint64_t>;

    VTKM_EXEC_CONT
    OutputType operator()(const MasksAndInputType& a,
                          const MasksAndInputType& b) const
    {
      return OutputType((a.first.first & a.second) +
                        (b.first.first & b.second),
                        (a.first.second & a.second) +
                        (b.first.second & b.second));
    }

    VTKM_EXEC_CONT
    OutputType operator()(const MasksAndInputType& a,
                          const OutputType& b) const
    {
      return OutputType((a.first.first & a.second) + b.first,
                        (a.first.second & a.second) + b.second);
    }

    VTKM_EXEC_CONT
    OutputType operator()(const OutputType& a,
                          const MasksAndInputType& b) const
    {
      return MaskAndAdd{}(b, a);
    }

    VTKM_EXEC_CONT
    OutputType operator()(const OutputType& a,
                          const OutputType& b) const
    {
      return OutputType(a.first + b.first, a.second + b.second);
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

  // It is important to remember that "bright" is first and "dark" is second
  auto brightAndDark = vtkm::cont::make_ArrayHandleZip(keysBright, keysDark);
  auto inputAndMasks = vtkm::cont::make_ArrayHandleZip(brightAndDark, input);

  using ResultType = vtkm::Pair<uint64_t, uint64_t>;
  const ResultType initialVal(0, 0);
  ResultType result =
    vtkm::cont::Algorithm::Reduce(inputAndMasks, initialVal, MaskAndAdd{});

  stemValues.bright = result.first;
  stemValues.dark = result.second;

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
