#include "image.h"

#include "config.h"
#include "mask.h"

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <vtkm/cont/ArrayHandleView.h>
#endif

#include <iostream>
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
  // Order is "input", "bright", and "dark"
  using InputType = vtkm::Vec<uint16_t, 3>;
  // Order is "bright" and "dark"
  using OutputType = vtkm::Pair<uint64_t, uint64_t>;

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a) const
  {
    return OutputType(a[0] & a[1], a[0] & a[2]);
  }

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a, const InputType& b) const
  {
    // Cast one of these to uint64_t to ensure no overflow on addition
    return OutputType(static_cast<uint64_t>(a[0] & a[1]) + (b[0] & b[1]),
                      static_cast<uint64_t>(a[0] & a[2]) + (b[0] & b[2]));
  }

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a, const OutputType& b) const
  {
    return OutputType((a[0] & a[1]) + b.first, (a[0] & a[2]) + b.second);
  }

  VTKM_EXEC_CONT
  OutputType operator()(const OutputType& a, const InputType& b) const
  {
    return MaskAndAdd{}(b, a);
  }

  VTKM_EXEC_CONT
  OutputType operator()(const OutputType& a, const OutputType& b) const
  {
    return OutputType(a.first + b.first, a.second + b.second);
  }
};
}

template <typename Storage>
STEMValues calculateSTEMValuesParallel(
  vtkm::cont::ArrayHandle<uint16_t, Storage> const& input,
  vtkm::cont::ArrayHandle<uint16_t> const& bright,
  vtkm::cont::ArrayHandle<uint16_t> const& dark, uint32_t imageNumber = -1)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;

  // It is important to remember that the order is "input", "bright", "dark"
  auto vector =
    vtkm::cont::make_ArrayHandleCompositeVector(input, bright, dark);

  using ResultType = vtkm::Pair<uint64_t, uint64_t>;
  const ResultType initialVal(0, 0);
  ResultType result =
    vtkm::cont::Algorithm::Reduce(vector, initialVal, MaskAndAdd{});

  stemValues.bright = result.first;
  stemValues.dark = result.second;

  return stemValues;
}
#endif

STEMImage createSTEMImage(BlockIterator& blockIt, int rows, int columns,  int innerRadius, int outerRadius)
{
  STEMImage image(rows, columns);

  if (blockIt.atEnd()) {
    cerr << "Error in " << __FUNCTION__ << ": no blocks to read!\n";
    return image;
  }

  const Block& block = *blockIt;

  // Get image size from first block
  auto detectorImageRows = block.header.rows;
  auto detectorImageColumns = block.header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  auto brightFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, 0, outerRadius);
  auto darkFieldMask = createAnnularMask(detectorImageRows, detectorImageColumns, innerRadius, outerRadius);

#ifdef VTKm
  // Only transfer the mask once.
  auto bright = vtkm::cont::make_ArrayHandle(brightFieldMask, numberOfPixels);
  auto dark = vtkm::cont::make_ArrayHandle(darkFieldMask, numberOfPixels);
#endif

  while(!blockIt.atEnd()) {
    auto data = (*blockIt).data.get();
#ifdef VTKm
    // Transfer the entire block of data at once.
    auto dataHandle = vtkm::cont::make_ArrayHandle(
      data, numberOfPixels * block.header.imagesInBlock);
#endif
    for(int i=0; i<block.header.imagesInBlock; i++) {
#ifdef VTKm
      // Use view to the array already transfered
      auto view = vtkm::cont::make_ArrayHandleView(
        dataHandle, i * numberOfPixels, numberOfPixels);
      auto stemValues = calculateSTEMValuesParallel(view, bright, dark);
#else
      auto stemValues = calculateSTEMValues(data, i*numberOfPixels, numberOfPixels,
                                            brightFieldMask, darkFieldMask);
#endif
      image.bright.data[block.header.imageNumbers[i]-1] = stemValues.bright;
      image.dark.data[block.header.imageNumbers[i]-1] = stemValues.dark;
    }

    // Move to the next block
    ++blockIt;
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
