#include "image.h"

#include "config.h"
#include "mask.h"

#include <ThreadPool.h>

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <vtkm/cont/ArrayHandleView.h>
#endif

#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

using namespace std;

namespace stempy {

template <typename T>
Image<T>::Image(uint32_t w, uint32_t h)
  : width(w), height(h), data(new T[w * h], std::default_delete<T[]>())
{ }

STEMImage::STEMImage(uint32_t width, uint32_t height)
{
  this->bright = Image<uint64_t>(width, height);
  this->dark = Image<uint64_t>(width, height);
  // Init values to zero
  std::fill(this->bright.data.get(), this->bright.data.get() + width*height, 0);
  std::fill(this->dark.data.get(), this->dark.data.get() + width*height, 0);
}

STEMValues calculateSTEMValues(const uint16_t data[], int offset,
                               int numberOfPixels, uint16_t brightFieldMask[],
                               uint16_t darkFieldMask[], uint32_t imageNumber)
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

// These should be ran by separate threads
namespace {
#ifdef VTKm
void _runCalculateSTEMValues(const uint16_t data[],
                             const vector<uint32_t>& imageNumbers,
                             uint32_t numberOfPixels,
                             const vtkm::cont::ArrayHandle<uint16_t>& bright,
                             const vtkm::cont::ArrayHandle<uint16_t>& dark,
                             STEMImage& image)
#else
void _runCalculateSTEMValues(const uint16_t data[],
                             const vector<uint32_t>& imageNumbers,
                             uint32_t numberOfPixels, uint16_t* brightFieldMask,
                             uint16_t* darkFieldMask, STEMImage& image)
#endif
{
#ifdef VTKm
  // Transfer the entire block of data at once.
  auto dataHandle =
    vtkm::cont::make_ArrayHandle(data, numberOfPixels * imageNumbers.size());
#endif
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
#ifdef VTKm
    // Use view to the array already transfered
    auto view = vtkm::cont::make_ArrayHandleView(dataHandle, i * numberOfPixels,
                                                 numberOfPixels);
    auto stemValues = calculateSTEMValuesParallel(view, bright, dark);
#else
    auto stemValues = calculateSTEMValues(
      data, i * numberOfPixels, numberOfPixels, brightFieldMask, darkFieldMask);
#endif
    image.bright.data[imageNumbers[i]] = stemValues.bright;
    image.dark.data[imageNumbers[i]] = stemValues.dark;
  }
}
} // end namespace

template <typename InputIt>
STEMImage createSTEMImage(InputIt first, InputIt last, int innerRadius,
                          int outerRadius, int rows, int columns, int centerX,
                          int centerY)
{
  if (first == last) {
    ostringstream msg;
    msg << "No blocks to read!";
    throw invalid_argument(msg.str());
  }

  // If we haven't been provided with rows and columns, try the header.
  if (rows == 0 || columns == 0) {
    rows = first->header.scanRows;
    columns = first->header.scanColumns;
  }

  // Raise an exception if we still don't have valid rows and columns
  if (rows <= 0 || columns <= 0) {
    ostringstream msg;
    msg << "No scan image size provided.";
    throw invalid_argument(msg.str());
  }

  STEMImage image(rows, columns);

  // Get image size from first block
  auto detectorImageRows = first->header.frameRows;
  auto detectorImageColumns = first->header.frameColumns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  auto brightFieldMask = createAnnularMask(
    detectorImageRows, detectorImageColumns, 0, outerRadius, centerX, centerY);
  auto darkFieldMask =
    createAnnularMask(detectorImageRows, detectorImageColumns, innerRadius,
                      outerRadius, centerX, centerY);

#ifdef VTKm
  // Only transfer the mask once.
  auto bright = vtkm::cont::make_ArrayHandle(brightFieldMask, numberOfPixels);
  auto dark = vtkm::cont::make_ArrayHandle(darkFieldMask, numberOfPixels);
#endif

  // Run the calculations in a thread pool while the data is read from
  // the disk in the main thread.
  // We benchmarked this on a 10 core computer, and typically found
  // 2 threads to be ideal.
  int numThreads = 2;
  ThreadPool pool(numThreads);

  // Populate the worker pool
  vector<future<void>> futures;
  for (; first != last; ++first) {
    // Move the block into the thread by copying... CUDA 10.1 won't allow
    // us to do something like "pool.enqueue([ b{ std::move(*first) }, ...])"
    Block b = std::move(*first);
#ifdef VTKm
    // Instead of calling _runCalculateSTEMValues directly, we use a
    // lambda so that we can explicity delete the block. Otherwise,
    // the block will not be deleted until the threads are destroyed.
    futures.emplace_back(
      pool.enqueue([b, numberOfPixels, &bright, &dark, &image]() mutable {
        _runCalculateSTEMValues(b.data.get(), b.header.imageNumbers,
                                numberOfPixels, bright, dark, image);
        // If we don't reset this, it won't get reset until the thread is
        // destroyed.
        b.data.reset();
      }));
#else
    futures.emplace_back(pool.enqueue(
      [b, numberOfPixels, brightFieldMask, darkFieldMask, &image]() mutable {
        _runCalculateSTEMValues(b.data.get(), b.header.imageNumbers,
                                numberOfPixels, brightFieldMask, darkFieldMask,
                                image);
        // If we don't reset this, it won't get reset until the thread is
        // destroyed.
        b.data.reset();
      }));
#endif
  }

  // Make sure all threads are finished before continuing
  for (auto& future : futures)
    future.get();

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}

vector<uint16_t> expandSparsifiedData(const vector<vector<uint32_t>>& data,
                                      size_t numPixels)
{
  vector<uint16_t> ret(data.size() * numPixels, 0);
  for (size_t i = 0; i < data.size(); ++i) {
    for (auto pos : data[i])
      ret[i * numPixels + pos] = 1;
  }

  return ret;
}

STEMImage createSTEMImageSparse(const vector<vector<uint32_t>>& sparseData,
                                int innerRadius, int outerRadius, int rows,
                                int columns, int frameRows, int frameColumns,
                                int centerX, int centerY)
{
  STEMImage image(rows, columns);

  auto brightFieldMask = createAnnularMask(frameRows, frameColumns, 0,
                                           outerRadius, centerX, centerY);
  auto darkFieldMask = createAnnularMask(frameRows, frameColumns, innerRadius,
                                         outerRadius, centerX, centerY);

  auto numberOfPixels = frameRows * frameColumns;
  vector<uint16_t> data = expandSparsifiedData(sparseData, numberOfPixels);

  size_t numImages = data.size() / numberOfPixels;
  vector<uint32_t> imageNumbers(numImages);
  std::iota(imageNumbers.begin(), imageNumbers.end(), 0);

#ifdef VTKm
  auto bright = vtkm::cont::make_ArrayHandle(brightFieldMask, numberOfPixels);
  auto dark = vtkm::cont::make_ArrayHandle(darkFieldMask, numberOfPixels);
  _runCalculateSTEMValues(data.data(), imageNumbers, numberOfPixels, bright,
                          dark, image);
#else
  _runCalculateSTEMValues(data.data(), imageNumbers, numberOfPixels,
                          brightFieldMask, darkFieldMask, image);
#endif

  delete[] brightFieldMask;
  delete[] darkFieldMask;

  return image;
}

template <typename InputIt>
Image<double> calculateAverage(InputIt first, InputIt last)
{
  auto detectorImageRows = first->header.frameRows;
  auto detectorImageColumns = first->header.frameColumns;
  auto numDetectorPixels = detectorImageRows * detectorImageColumns;
  Image<double> image(detectorImageRows, detectorImageColumns);

  std::fill(image.data.get(), image.data.get() + numDetectorPixels, 0.0);
  uint64_t numberOfImages = 0;
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto blockData = block.data.get();
    numberOfImages += block.header.imagesInBlock;
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto numberOfPixels = block.header.frameRows * block.header.frameColumns;
      for (unsigned j = 0; j < numberOfPixels; j++) {
        image.data[j] += blockData[i*numberOfPixels+j];
      }
    }
  }

  for (unsigned i = 0; i < detectorImageRows * detectorImageColumns; i++) {
    image.data[i] /= numberOfImages;
  }

  return image;
}

// Instantiate the ones that can be used
template STEMImage createSTEMImage(StreamReader::iterator first,
                                   StreamReader::iterator last, int rows,
                                   int columns, int innerRadius,
                                   int outerRadius, int centerX, int centerY);
template STEMImage createSTEMImage(vector<Block>::iterator first,
                                   vector<Block>::iterator last, int rows,
                                   int columns, int innerRadius,
                                   int outerRadius, int centerX, int centerY);

template Image<double> calculateAverage(StreamReader::iterator first,
                                        StreamReader::iterator last);
template Image<double> calculateAverage(vector<Block>::iterator first,
                                        vector<Block>::iterator last);
}
