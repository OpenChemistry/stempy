#include "image.h"
#include "python/pyreader.h"

#include "config.h"
#include "electron.h"
#include "mask.h"

#include <ThreadPool.h>

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <vtkm/cont/ArrayHandleView.h>
#include <vtkm/cont/AtomicArray.h>
#include <vtkm/cont/Invoker.h>

template <typename T>
using ArrayHandleView = vtkm::cont::ArrayHandleView<vtkm::cont::ArrayHandle<T>>;

#endif

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

using std::begin;
using std::end;
using std::future;
using std::invalid_argument;
using std::ostringstream;
using std::vector;

namespace stempy {

STEMValues calculateSTEMValues(const uint16_t data[], uint64_t offset,
                               uint32_t numberOfPixels, uint16_t mask[],
                               uint32_t imageNumber)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;
  for (size_t i = 0; i < numberOfPixels; i++) {
    auto value = data[offset + i];
    stemValues.data += value & mask[i];
  }

  return stemValues;
}

template <typename T>
RadialSum<T>::RadialSum(Dimensions2D dims, uint32_t r)
  : dimensions(dims), radii(r),
    data(new T[dims.first * dims.second * r], std::default_delete<T[]>())
{
  std::fill(this->data.get(),
            this->data.get() + dims.first * dims.second * radii, 0);
}


#ifdef VTKm
namespace {
struct MaskAndAdd
{
  // Order is "input", "mask"
  using InputType = vtkm::Vec<uint16_t, 2>;
  using OutputType = uint64_t;

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a) const { return a[0] & a[1]; }

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a, const InputType& b) const
  {
    // Cast one of these to OutputType to ensure no overflow on addition
    return static_cast<OutputType>(a[0] & a[1]) + (b[0] & b[1]);
  }

  VTKM_EXEC_CONT
  OutputType operator()(const InputType& a, const OutputType& b) const
  {
    return (a[0] & a[1]) + b;
  }

  VTKM_EXEC_CONT
  OutputType operator()(const OutputType& a, const InputType& b) const
  {
    return MaskAndAdd{}(b, a);
  }

  VTKM_EXEC_CONT
  OutputType operator()(const OutputType& a, const OutputType& b) const
  {
    return a + b;
  }
};
}

template <typename Storage>
STEMValues calculateSTEMValuesParallel(
  vtkm::cont::ArrayHandle<uint16_t, Storage> const& input,
  vtkm::cont::ArrayHandle<uint16_t> const& mask, uint32_t imageNumber = -1)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;

  // It is important to remember that the order is "input", "mask"
  auto vector = vtkm::cont::make_ArrayHandleCompositeVector(input, mask);

  using ResultType = uint64_t;
  const ResultType initialVal(0);
  stemValues.data =
    vtkm::cont::Algorithm::Reduce(vector, initialVal, MaskAndAdd{});

  return stemValues;
}
#endif

// These should be ran by separate threads
namespace {
#ifdef VTKm
void _runCalculateSTEMValues(const uint16_t data[],
                             const vector<uint32_t>& imageNumbers,
                             uint32_t numberOfPixels,
                             const vtkm::cont::ArrayHandle<uint16_t>& mask,
                             STEMImage& image)
#else
void _runCalculateSTEMValues(const uint16_t data[],
                             const vector<uint32_t>& imageNumbers,
                             uint32_t numberOfPixels, uint16_t* mask,
                             STEMImage& image)
#endif
{
#ifdef VTKm
  // Transfer the entire block of data at once.
  auto dataHandle =
    vtkm::cont::make_ArrayHandle(data, numberOfPixels * imageNumbers.size());
#endif
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
    // We need to ensure we are using uint64_t to prevent overflow.
    auto offset = static_cast<uint64_t>(i) * numberOfPixels;
#ifdef VTKm
    // Use view to the array already transfered
    // Note: We need to ensure the offset is pass as a uint64 to prevent
    // overflow.
    auto view = vtkm::cont::make_ArrayHandleView(dataHandle, offset,
                                                 numberOfPixels);
    auto stemValues = calculateSTEMValuesParallel(view, mask);
#else
    auto stemValues =
      calculateSTEMValues(data, offset,
                          numberOfPixels, mask, imageNumbers[i]);
#endif
    image.data.get()[imageNumbers[i]] = stemValues.data;
  }
}
} // end namespace

template <typename InputIt>
vector<STEMImage> createSTEMImages(InputIt first, InputIt last,
                                   const vector<int>& innerRadii,
                                   const vector<int>& outerRadii,
                                   Dimensions2D scanDimensions,
                                   Coordinates2D center)
{
  if (first == last) {
    ostringstream msg;
    msg << "No blocks to read!";
    throw invalid_argument(msg.str());
  }

  if (innerRadii.empty() || outerRadii.empty()) {
    ostringstream msg;
    msg << "innerRadii or outerRadii are empty!";
    throw invalid_argument(msg.str());
  }

  if (innerRadii.size() != outerRadii.size()) {
    ostringstream msg;
    msg << "innerRadii and outerRadii are not the same size!";
    throw invalid_argument(msg.str());
  }

  // If we haven't been provided with width and height, try the header.
  if (scanDimensions.first == 0 || scanDimensions.second == 0) {
    scanDimensions = first->header.scanDimensions;
  }

  // Raise an exception if we still don't have valid width and height
  if (scanDimensions.first <= 0 || scanDimensions.second <= 0) {
    ostringstream msg;
    msg << "No scan image size provided.";
    throw invalid_argument(msg.str());
  }

  vector<STEMImage> images;
  for (const auto& r : innerRadii) {
    (void)r;
    images.push_back(STEMImage(scanDimensions));
  }
  // Get image size from first block
  auto frameDimensions = first->header.frameDimensions;
  auto numberOfPixels = frameDimensions.first * frameDimensions.second;

  vector<uint16_t*> masks;

#ifdef VTKm
  // Only transfer the masks once
  vector<vtkm::cont::ArrayHandle<uint16_t>> maskHandles;
#endif

  for (size_t i = 0; i < innerRadii.size(); ++i) {
    masks.push_back(
      createAnnularMask(frameDimensions, innerRadii[i], outerRadii[i], center));

#ifdef VTKm
    maskHandles.push_back(
      vtkm::cont::make_ArrayHandle(masks.back(), numberOfPixels));
#endif
  }

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
    auto b = std::move(*first);

    for (size_t i = 0; i < masks.size(); ++i) {
      auto& image = images[i];
#ifdef VTKm
      const auto& maskHandle = maskHandles[i];

      // Instead of calling _runCalculateSTEMValues directly, we use a
      // lambda so that we can explicity delete the block. Otherwise,
      // the block will not be deleted until the threads are destroyed.
      futures.emplace_back(
        pool.enqueue([b, numberOfPixels, &maskHandle, &image]() mutable {
          _runCalculateSTEMValues(b.data.get(), b.header.imageNumbers,
                                  numberOfPixels, maskHandle, image);
          // If we don't reset this, it won't get reset until the thread is
          // destroyed.
          b.data.reset();
        }));
#else
      const auto& mask = masks[i];
      futures.emplace_back(
        pool.enqueue([b, numberOfPixels, mask, &image]() mutable {
          _runCalculateSTEMValues(b.data.get(), b.header.imageNumbers,
                                  numberOfPixels, mask, image);
          // If we don't reset this, it won't get reset until the thread is
          // destroyed.
          b.data.reset();
        }));
#endif
    }
  }

  // Make sure all threads are finished before continuing
  for (auto& future : futures)
    future.get();

  for (const auto* p : masks)
    delete[] p;

  return images;
}

// returns the binding result
std::vector<double> getContainer(const STEMImage& inImage, const int numBins)
{
  // information about input STEMImage
  auto scanDimensions = inImage.dimensions;
  auto curData = inImage.data;

  auto result =
    std::minmax_element(curData.get(), curData.get() + scanDimensions.first *
                                                         scanDimensions.second);
  double min = *result.first;
  double max = *result.second;

  // the "length" of each slot of the container
  double length = (max - min) / numBins;

  std::vector<double> container;
  // push all the intermediate values
  for (int i = 0; i <= numBins; i++) {
    container.push_back(min + i * length);
  }

  return container;
}

// function that computes histogram for all the STEM images
// each histogram is a vector<int>
std::vector<int> createSTEMHistogram(const STEMImage& inImage,
                                     const int numBins,
                                     const std::vector<double> bins)
{
  // initialize output
  std::vector<int> frequencies(numBins, 0);

  // STEMImage info
  auto scanDimensions = inImage.dimensions;
  auto curData = inImage.data.get();

  // get a histrogram
  for (uint32_t i = 0; i < scanDimensions.first * scanDimensions.second; ++i) {
    auto value = curData[i];
    // check which bin it belongs to
    for (int j = 0; j < numBins; j++) {
      if (value >= bins[j] && value < bins[j + 1]) {
        ++frequencies[j];
      }
    }
    // the max value is put in the last slot
    if (value == bins[numBins]) {
      ++frequencies[numBins - 1];
    }
  }

  return frequencies;
}

vector<STEMImage> createSTEMImages(const ElectronCountedData& data,
                                   const vector<int>& innerRadii,
                                   const vector<int>& outerRadii,
                                   Coordinates2D center)
{
  return createSTEMImages(data.data, innerRadii, outerRadii,
                          data.scanDimensions, data.frameDimensions, center);
}

template <typename InputIt>
Image<double> calculateAverage(InputIt first, InputIt last)
{
  auto frameDimensions = first->header.frameDimensions;
  auto numDetectorPixels = frameDimensions.first * frameDimensions.second;
  Image<double> image(frameDimensions);

  std::fill(image.data.get(), image.data.get() + numDetectorPixels, 0.0);
  uint64_t numberOfImages = 0;
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto blockData = block.data.get();
    numberOfImages += block.header.imagesInBlock;
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto numberOfPixels = block.header.frameDimensions.first *
                            block.header.frameDimensions.second;
      for (unsigned j = 0; j < numberOfPixels; j++) {
        image.data.get()[j] += blockData[i * numberOfPixels + j];
      }
    }
  }

  for (unsigned i = 0; i < frameDimensions.first * frameDimensions.second;
       i++) {
    image.data.get()[i] /= numberOfImages;
  }

  return image;
}

double inline distance(int x1, int y1, int x2, int y2) {
  return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0));
}

void radialSumFrame(Coordinates2D center, const uint16_t data[],
                    uint64_t offset, Dimensions2D frameDimensions,
                    int imageNumber, RadialSum<uint64_t>& radialSum)
{
  auto numberOfPixels = frameDimensions.first * frameDimensions.second;
  for (uint32_t i = 0; i < numberOfPixels; ++i) {
    auto x = i % frameDimensions.first;
    auto y = i / frameDimensions.first;
    auto radius =
      static_cast<int>(std::ceil(distance(x, y, center.first, center.second)));
    // Use compiler intrinsic to ensure atomic add
    auto address =
      radialSum.data.get() +
      radius * radialSum.dimensions.first * radialSum.dimensions.second +
      imageNumber;
    __sync_fetch_and_add(address, data[offset + i]);
  }
}

namespace {

#ifdef VTKm
struct RadialSumWorklet : public vtkm::worklet::WorkletMapField
{
  vtkm::Vec<int, 2> m_center;
  int m_width;
  int m_imageNumber;
  uint32_t m_numberOfScanPositions;

  RadialSumWorklet(vtkm::Vec<int, 2> center, int width,
                   int imageNumber, uint32_t numberOfScanPositions)
    : m_center(center), m_width(width), m_imageNumber(imageNumber),
      m_numberOfScanPositions(numberOfScanPositions){};

  using ControlSignature = void(FieldIn, AtomicArrayInOut);

  using ExecutionSignature = void(_1, _2, WorkIndex);

  template <typename AtomicInterface>
  VTKM_EXEC void operator()(const vtkm::UInt16& value,
                            AtomicInterface& radialSum, vtkm::Id i) const
  {
    auto x = i % m_width;
    auto y = i / m_width;
    auto radius =
      static_cast<int>(std::ceil(distance(x, y, m_center[0], m_center[1])));

    radialSum.Add(radius * m_numberOfScanPositions + m_imageNumber, value);
  }
};

void radialSumFrame(const vtkm::Vec<int, 2>& center,
                    const ArrayHandleView<vtkm::UInt16>& data,
                    int frameWidth, int imageNumber,
                    uint32_t numberOfScanPositions,
                    vtkm::cont::ArrayHandle<vtkm::Int64>& radialSum)
{
  vtkm::cont::Invoker invoke;
  invoke(RadialSumWorklet{ center, frameWidth, imageNumber, numberOfScanPositions },
         data, radialSum);
}

void radialSumFrames(Coordinates2D center, const uint16_t data[],
                     int frameWidth, std::vector<uint32_t>& imageNumbers,
                     uint32_t numberOfPixels, uint32_t numberOfScanPositions,
                     vtkm::cont::ArrayHandle<vtkm::Int64>& radialSum)
{
  vtkm::Vec<int, 2> centerVec = { center.first, center.second };
  auto dataHandle =
    vtkm::cont::make_ArrayHandle(data, numberOfPixels * imageNumbers.size());
  // Use view to the array already transfered
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
    // We need to ensure we are using uint64_t to prevent overflow.
    auto offset = static_cast<uint64_t>(i) * numberOfPixels;
    auto view =
      vtkm::cont::make_ArrayHandleView(dataHandle, offset, numberOfPixels);

    radialSumFrame(centerVec, view, frameWidth, imageNumbers[i],
                   numberOfScanPositions, radialSum);
  }
}

#else
void radialSumFrames(Coordinates2D center, const uint16_t data[],
                     Dimensions2D frameDimensions,
                     std::vector<uint32_t>& imageNumbers,
                     uint32_t numberOfPixels, RadialSum<uint64_t>& radialSum)
{
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
    // We need to ensure we are using int64_t to prevent overflow.
    auto offset = static_cast<int64_t>(i) * numberOfPixels;
    auto imageNumber = imageNumbers[i];
    radialSumFrame(center, data, offset, frameDimensions, imageNumber,
                   radialSum);
  }
}
#endif
}

template <typename InputIt>
RadialSum<uint64_t> radialSum(InputIt first, InputIt last,
                              Dimensions2D scanDimensions, Coordinates2D center)
{
  if (first == last) {
    ostringstream msg;
    msg << "No blocks to read!";
    throw invalid_argument(msg.str());
  }

  // If we haven't been provided with width and height, try the header.
  if (scanDimensions.first == 0 || scanDimensions.second == 0) {
    scanDimensions = first->header.scanDimensions;
  }

  // Raise an exception if we still don't have valid width and height
  if (scanDimensions.first <= 0 || scanDimensions.second <= 0) {
    ostringstream msg;
    msg << "No scan image size provided.";
    throw invalid_argument(msg.str());
  }

  // Get image size from first block
  auto frameDimensions = first->header.frameDimensions;
  auto numberOfPixels = frameDimensions.first * frameDimensions.second;

  // Default the center if necessary
  if (center.first < 0)
    center.first = static_cast<int>(std::round(frameDimensions.first / 2.0));

  if (center.second < 0)
    center.second = static_cast<int>(std::round(frameDimensions.second / 2.0));

  // Calculate the maximum possible radius for the frame, the maximum distance
  // from all four corners
  double max = 0.0;
  for(int x=0; x<2; x++) {
    for(int y=0; y<2; y++) {
      auto dist =
        distance(x * frameDimensions.first, y * frameDimensions.second,
                 center.first, center.second);
      if (dist > max) {
        max = dist;
      }
    }
  }

  int maxRadius = static_cast<int>(std::ceil(max));

  // Run the calculations in a thread pool while the data is read from
  // the disk in the main thread.
  // We benchmarked this on a 10 core computer, and typically found
  // 2 threads to be ideal.
  int numThreads = 2;
  ThreadPool pool(numThreads);
  RadialSum<uint64_t> radialSum(scanDimensions, maxRadius + 1);

#ifdef VTKm
  // We need the reinterpret_cast as vtkm currently doesn't support atomic
  // access for uint64.
  auto radialSumHandle = vtkm::cont::make_ArrayHandle(
    reinterpret_cast<vtkm::Int64*>(radialSum.data.get()),
    radialSum.radii * radialSum.dimensions.first * radialSum.dimensions.second);
#endif

  // Populate the worker pool
  vector<future<void>> futures;
  for (; first != last; ++first) {
    // Move the block into the thread by copying... CUDA 10.1 won't allow
    // us to do something like "pool.enqueue([ b{ std::move(*first) }, ...])"
    auto b = std::move(*first);
    // Instead of calling _runCalculateSTEMValues directly, we use a
    // lambda so that we can explicity delete the block. Otherwise,
    // the block will not be deleted until the threads are destroyed.
#ifdef VTKm
    auto numberOfScanPositions =
      radialSum.dimensions.first * radialSum.dimensions.second;
    futures.emplace_back(
      pool.enqueue([b, numberOfPixels, center, frameDimensions,
                    &radialSumHandle, numberOfScanPositions]() mutable {
        radialSumFrames(center, b.data.get(), frameDimensions.first,
                        b.header.imageNumbers, numberOfPixels,
                        numberOfScanPositions, radialSumHandle);
        // If we don't reset this, it won't get reset until the thread is
        // destroyed.
        b.data.reset();
      }));
#else
    futures.emplace_back(pool.enqueue(
      [b, numberOfPixels, center, frameDimensions, &radialSum]() mutable {
        radialSumFrames(center, b.data.get(), frameDimensions,
                        b.header.imageNumbers, numberOfPixels, radialSum);
        // If we don't reset this, it won't get reset until the thread is
        // destroyed.
        b.data.reset();
      }));
#endif
  }

  // Make sure all threads are finished before continuing
  for (auto& future : futures)
    future.get();

  return radialSum;
}

template <typename InputIt>
Image<double> maximumDiffractionPattern(InputIt first, InputIt last,
                                        const Image<float>& darkreference)
{
  auto frameDimensions = first->header.frameDimensions;
  auto numDetectorPixels = frameDimensions.first * frameDimensions.second;
  Image<double> maxDiffPattern(frameDimensions);

  std::fill(maxDiffPattern.data.get(),
            maxDiffPattern.data.get() + numDetectorPixels, 0);
  uint64_t numberOfImages = 0;
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto blockData = block.data.get();
    numberOfImages += block.header.imagesInBlock;
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto numberOfPixels = block.header.frameDimensions.first *
                            block.header.frameDimensions.second;
      for (unsigned j = 0; j < numberOfPixels; j++) {
        if (blockData[i * numberOfPixels + j] > maxDiffPattern.data.get()[j]) {
          maxDiffPattern.data.get()[j] = blockData[i * numberOfPixels + j];
        }
      }
    }
  }

  // If we have been given a darkreference substract it
  if (darkreference.dimensions.first > 0) {
    for (unsigned i = 0; i < numDetectorPixels; i++) {
      maxDiffPattern.data.get()[i] -= darkreference.data.get()[i];
    }
  }

  return maxDiffPattern;
}

template <typename InputIt>
Image<double> maximumDiffractionPattern(InputIt first, InputIt last)
{
  // Create empty dark reference
  Image<float> dark;

  return maximumDiffractionPattern(first, last, dark);
}

// Instantiate the ones that can be used
template vector<STEMImage> createSTEMImages<StreamReader::iterator>(
  StreamReader::iterator first, StreamReader::iterator last,
  const vector<int>& innerRadii, const vector<int>& outerRadii,
  Dimensions2D scanDimensions, Coordinates2D center);

template vector<STEMImage> createSTEMImages<PyReader::iterator>(
  PyReader::iterator first, PyReader::iterator last,
  const vector<int>& innerRadii, const vector<int>& outerRadii,
  Dimensions2D scanDimensions, Coordinates2D center);

template vector<STEMImage> createSTEMImages<vector<Block>::iterator>(
  vector<Block>::iterator first, vector<Block>::iterator last,
  const vector<int>& innerRadii, const vector<int>& outerRadii,
  Dimensions2D scanDimensions, Coordinates2D center);

template vector<STEMImage> createSTEMImages<SectorStreamReader::iterator>(
  SectorStreamReader::iterator first, SectorStreamReader::iterator last,
  const vector<int>& innerRadii, const vector<int>& outerRadii,
  Dimensions2D scanDimensions, Coordinates2D center);

template Image<double> calculateAverage(StreamReader::iterator first,
                                        StreamReader::iterator last);
template Image<double> calculateAverage(vector<Block>::iterator first,
                                        vector<Block>::iterator last);
template Image<double> calculateAverage(SectorStreamReader::iterator first,
                                        SectorStreamReader::iterator last);
template Image<double> calculateAverage(PyReader::iterator first,
                                        PyReader::iterator last);

template RadialSum<uint64_t> radialSum(StreamReader::iterator first,
                                       StreamReader::iterator last,
                                       Dimensions2D scanDimensions,
                                       Coordinates2D center);
template RadialSum<uint64_t> radialSum(vector<Block>::iterator,
                                       vector<Block>::iterator last,
                                       Dimensions2D scanDimensions,
                                       Coordinates2D center);
template RadialSum<uint64_t> radialSum(SectorStreamReader::iterator first,
                                       SectorStreamReader::iterator last,
                                       Dimensions2D scanDimensions,
                                       Coordinates2D center);
template RadialSum<uint64_t> radialSum(PyReader::iterator first,
                                       PyReader::iterator last,
                                       Dimensions2D scanDimensions,
                                       Coordinates2D center);

template Image<double> maximumDiffractionPattern(
  StreamReader::iterator first, StreamReader::iterator last,
  const Image<float>& darkreference);
template Image<double> maximumDiffractionPattern(
  vector<Block>::iterator first, vector<Block>::iterator last,
  const Image<float>& darkreference);
template Image<double> maximumDiffractionPattern(
  SectorStreamReader::iterator first, SectorStreamReader::iterator last,
  const Image<float>& darkreference);
template Image<double> maximumDiffractionPattern(
  PyReader::iterator first, PyReader::iterator last,
  const Image<float>& darkreference);
template Image<double> maximumDiffractionPattern(StreamReader::iterator first,
                                                 StreamReader::iterator last);
template Image<double> maximumDiffractionPattern(vector<Block>::iterator first,
                                                 vector<Block>::iterator last);
template Image<double> maximumDiffractionPattern(
  SectorStreamReader::iterator first, SectorStreamReader::iterator last);
template Image<double> maximumDiffractionPattern(
  PyReader::iterator first, PyReader::iterator last);
}
