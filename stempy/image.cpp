#include "image.h"
#include "python/reader_h5.h"

#include "config.h"
#include "mask.h"

#include <ThreadPool.h>

#ifdef VTKm
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <vtkm/cont/ArrayHandleView.h>
#include <vtkm/cont/AtomicArray.h>
#include <vtkm/worklet/Invoker.h>

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

template <typename T>
Image<T>::Image(uint32_t w, uint32_t h)
  : width(w), height(h), data(new T[w * h], std::default_delete<T[]>())
{
  std::fill(this->data.get(), this->data.get() + width * height, 0);
}

STEMValues calculateSTEMValues(const uint16_t data[], int offset,
                               int numberOfPixels, uint16_t mask[],
                               uint32_t imageNumber)
{
  STEMValues stemValues;
  stemValues.imageNumber = imageNumber;
  for (int i=0; i<numberOfPixels; i++) {

     std::cout<<"---debug index: "<< i << std::endl;
    auto value = data[offset + i];
     std::cout<<"---value---"<< value << std::endl;
    stemValues.data += value & mask[i];

  }

  return stemValues;
}

template <typename T>
RadialSum<T>::RadialSum(uint32_t w, uint32_t h, uint32_t r)
  : width(w), height(h), radii(r), data(new T[w * h * r], std::default_delete<T[]>())
{
  std::fill(this->data.get(), this->data.get() + width * height * radii, 0);
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
  
  std::cout<<"debug start to execute _runCalculateSTEMValues" <<std::endl;

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
    auto stemValues = calculateSTEMValuesParallel(view, mask);
#else
    std::cout<< " start to execute calculateSTEMValues " <<std::endl;
    auto stemValues =
      calculateSTEMValues(data, i * numberOfPixels, numberOfPixels, mask);
    std::cout<< " ok to execute calculateSTEMValues " <<std::endl;

#endif
    image.data[imageNumbers[i]] = stemValues.data;
  }
}
} // end namespace

template <typename InputIt, typename BlockType>
vector<STEMImage> createSTEMImages(InputIt first, InputIt last,
                                   vector<int> innerRadii,
                                   vector<int> outerRadii, int width,
                                   int height, int centerX, int centerY)
{
  std::cout<<"debug createImage, start function"<<std::endl;
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
  if (width == 0 || height == 0) {
    width = first->header.scanWidth;
    height = first->header.scanHeight;
  }

  // Raise an exception if we still don't have valid width and height
  if (width <= 0 || height <= 0) {
    ostringstream msg;
    msg << "No scan image size provided.";
    throw invalid_argument(msg.str());
  }

  vector<STEMImage> images;
  for (const auto& r : innerRadii) {
    (void)r;
    images.push_back(STEMImage(width, height));
  }
  // Get image size from first block
  auto frameWidth = first->header.frameWidth;
  auto frameHeight = first->header.frameHeight;
  auto numberOfPixels = frameWidth * frameHeight;

  vector<uint16_t*> masks;

#ifdef VTKm
  // Only transfer the masks once
  vector<vtkm::cont::ArrayHandle<uint16_t>> maskHandles;
#endif

  for (size_t i = 0; i < innerRadii.size(); ++i) {
    masks.push_back(createAnnularMask(frameWidth, frameHeight, innerRadii[i],
                                      outerRadii[i], centerX, centerY));

#ifdef VTKm
    maskHandles.push_back(
      vtkm::cont::make_ArrayHandle(masks.back(), numberOfPixels));
#endif
  }

  // Run the calculations in a thread pool while the data is read from
  // the disk in the main thread.
  // We benchmarked this on a 10 core computer, and typically found
  // 2 threads to be ideal.
  int numThreads = 1;
  ThreadPool pool(numThreads);

  // Populate the worker pool
  vector<future<void>> futures;
  std::cout<<"debug createImage, start for loop"<<std::endl;
  for (; first != last; ++first) {
    // Move the block into the thread by copying... CUDA 10.1 won't allow
    // us to do something like "pool.enqueue([ b{ std::move(*first) }, ...])"
    BlockType b = std::move(*first);
    std::cout<<"debug createImage, ok to get block"<<std::endl;


    for (size_t i = 0; i < masks.size(); ++i) {
      auto& image = images[i];
#ifdef VTKm
      std::cout<<"vtk-m is used"<<std::endl;
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
        std::cout<<"start to put block into lambda"<<std::endl;

            std::cout<< "check data in block"<<std::endl;

  for (int i = 0; i < 10; i++) {
    std::cout << *(b.data.get() + i) << std::endl;
  }

      std::cout<< "check header in block"<<std::endl;
 for (int i = 0; i < 10; i++) {
      std::cout<<"image number " << b.header.imageNumbers[i] <<std::endl;
}

      std::cout<< "check numberOfPixels "<< numberOfPixels <<std::endl;
      std::cout<< "check mask "<< *mask <<std::endl;


      futures.emplace_back(
        pool.enqueue([b, numberOfPixels, mask, &image]() mutable {
          _runCalculateSTEMValues(b.data.get(), b.header.imageNumbers,
                                  numberOfPixels, mask, image);
          // If we don't reset this, it won't get reset until the thread is
          // destroyed.
          b.data.reset();
        }));

        std::cout<<"ok to put block into lambda1"<<std::endl;

#endif
    }
  }

  std::cout<<"ok to put block into lambda2"<<std::endl;

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
  int width = inImage.width;
  int height = inImage.height;
  auto curData = inImage.data;

  auto result =
    std::minmax_element(curData.get(), curData.get() + width * height);
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
  int width = inImage.width;
  int height = inImage.height;
  auto curData = inImage.data;

  // get a histrogram
  for (int i = 0; i < width * height; i++) {
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

vector<STEMImage> createSTEMImagesSparse(
  const vector<vector<uint32_t>>& sparseData, vector<int> innerRadii,
  vector<int> outerRadii, int width, int height, int frameWidth,
  int frameHeight, int centerX, int centerY)
{
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

  auto numberOfPixels = frameWidth * frameHeight;

  vector<STEMImage> images;
  for (const auto& r : innerRadii) {
    (void)r;
    images.push_back(STEMImage(width, height));
  }

  vector<uint16_t*> masks;

#ifdef VTKm
  // Only transfer the masks once
  vector<vtkm::cont::ArrayHandle<uint16_t>> maskHandles;
#endif

  for (size_t i = 0; i < innerRadii.size(); ++i) {
    masks.push_back(createAnnularMask(frameWidth, frameHeight, innerRadii[i],
                                      outerRadii[i], centerX, centerY));
#ifdef VTKm
    maskHandles.push_back(
      vtkm::cont::make_ArrayHandle(masks.back(), numberOfPixels));
#endif
  }

  vector<uint16_t> data = expandSparsifiedData(sparseData, numberOfPixels);

  size_t numImages = data.size() / numberOfPixels;
  vector<uint32_t> imageNumbers(numImages);
  std::iota(imageNumbers.begin(), imageNumbers.end(), 0);

  for (size_t i = 0; i < masks.size(); ++i) {
#ifdef VTKm
    _runCalculateSTEMValues(data.data(), imageNumbers, numberOfPixels,
                            maskHandles[i], images[i]);
#else
    _runCalculateSTEMValues(data.data(), imageNumbers, numberOfPixels, masks[i],
                            images[i]);
#endif
  }

  for (auto* p : masks)
    delete[] p;

  return images;
}

template <typename InputIt>
Image<double> calculateAverage(InputIt first, InputIt last)
{
  auto frameWidth = first->header.frameWidth;
  auto frameHeight = first->header.frameHeight;
  auto numDetectorPixels = frameWidth*frameHeight;
  Image<double> image(frameWidth, frameHeight);

  std::fill(image.data.get(), image.data.get() + numDetectorPixels, 0.0);
  uint64_t numberOfImages = 0;
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto blockData = block.data.get();
    numberOfImages += block.header.imagesInBlock;
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto numberOfPixels = block.header.frameHeight * block.header.frameWidth;
      for (unsigned j = 0; j < numberOfPixels; j++) {
        image.data[j] += blockData[i*numberOfPixels+j];
      }
    }
  }

  for (unsigned i = 0; i < frameHeight * frameWidth; i++) {
    image.data[i] /= numberOfImages;
  }

  return image;
}

double inline distance(int x1, int y1, int x2, int y2) {
  return sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0));
}

void radialSumFrame(int centerX, int centerY, const uint16_t data[],
    int offset, int frameWidth, int frameHeight, int imageNumber, RadialSum<uint64_t>& radialSum)
{
  auto numberOfPixels = frameWidth*frameHeight;
  for (int i=0; i< numberOfPixels; i++) {
    auto x = i % frameWidth;
    auto y = i / frameWidth;
    auto radius = static_cast<int>(
        std::ceil(
            distance(x, y, centerX, centerY)
        )
    );
    // Use compiler intrinsic to ensure atomic add
    auto address = radialSum.data.get() + radius*radialSum.width*radialSum.height + imageNumber;
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
  vtkm::worklet::Invoker invoke;
  invoke(RadialSumWorklet{ center, frameWidth, imageNumber, numberOfScanPositions },
         data, radialSum);
}

void radialSumFrames(int centerX, int centerY, const uint16_t data[],
                     int frameWidth, std::vector<uint32_t>& imageNumbers,
                     uint32_t numberOfPixels, uint32_t numberOfScanPositions,
                     vtkm::cont::ArrayHandle<vtkm::Int64>& radialSum)
{
  vtkm::Vec<int, 2> center = { centerX, centerY };
  auto dataHandle =
    vtkm::cont::make_ArrayHandle(data, numberOfPixels * imageNumbers.size());
  // Use view to the array already transfered
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
    auto offset = i * numberOfPixels;
    auto view =
      vtkm::cont::make_ArrayHandleView(dataHandle, offset, numberOfPixels);

    radialSumFrame(center, view, frameWidth, imageNumbers[i], numberOfScanPositions,
                   radialSum);
  }
}

#else
void radialSumFrames(int centerX, int centerY, const uint16_t data[],
                     int frameWidth, int frameHeight,
                     std::vector<uint32_t>& imageNumbers,
                     uint32_t numberOfPixels, RadialSum<uint64_t>& radialSum)
{
  for (unsigned i = 0; i < imageNumbers.size(); ++i) {
    auto offset = i*numberOfPixels;
    auto imageNumber = imageNumbers[i];
    radialSumFrame(centerX, centerY, data, offset,
        frameWidth, frameHeight, imageNumber, radialSum);
  }
}
#endif
}

template <typename InputIt>
RadialSum<uint64_t> radialSum(InputIt first, InputIt last, int scanWidth, int scanHeight,
      int centerX, int centerY)
{
  if (first == last) {
    ostringstream msg;
    msg << "No blocks to read!";
    throw invalid_argument(msg.str());
  }

  // If we haven't been provided with width and height, try the header.
  if (scanWidth == 0 || scanHeight == 0) {
    scanWidth = first->header.scanWidth;
    scanHeight = first->header.scanHeight;
  }

  // Raise an exception if we still don't have valid width and height
  if (scanWidth <= 0 || scanHeight <= 0) {
    ostringstream msg;
    msg << "No scan image size provided.";
    throw invalid_argument(msg.str());
  }

  // Get image size from first block
  auto frameWidth = first->header.frameWidth;
  auto frameHeight = first->header.frameHeight;
  auto numberOfPixels = frameWidth * frameHeight;

  // Default the center if necessary
  if (centerX < 0)
    centerX = static_cast<int>(std::round(frameWidth / 2.0));

  if (centerY < 0)
    centerY = static_cast<int>(std::round(frameHeight / 2.0));

  // Calculate the maximum possible radius for the frame, the maximum distance
  // from all four corners
  double max = 0.0;
  for(int x=0; x<2; x++) {
    for(int y=0; y<2; y++) {
      auto dist = distance(x*frameWidth, y*frameHeight, centerX, centerY);
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
  RadialSum<uint64_t> radialSum(scanWidth, scanHeight, maxRadius+1);

#ifdef VTKm
  // We need the reinterpret_cast as vtkm currently doesn't support atomic
  // access for uint64.
  auto radialSumHandle = vtkm::cont::make_ArrayHandle(
    reinterpret_cast<vtkm::Int64*>(radialSum.data.get()),
    radialSum.radii * radialSum.width * radialSum.height);
#endif

  // Populate the worker pool
  vector<future<void>> futures;
  for (; first != last; ++first) {
    // Move the block into the thread by copying... CUDA 10.1 won't allow
    // us to do something like "pool.enqueue([ b{ std::move(*first) }, ...])"
    Block b = std::move(*first);
    // Instead of calling _runCalculateSTEMValues directly, we use a
    // lambda so that we can explicity delete the block. Otherwise,
    // the block will not be deleted until the threads are destroyed.
#ifdef VTKm
    auto numberOfScanPositions = radialSum.width * radialSum.height;
    futures.emplace_back(pool.enqueue(
      [b, numberOfPixels, centerX, centerY, frameWidth,
       &radialSumHandle, numberOfScanPositions]() mutable {
        radialSumFrames(centerX, centerY, b.data.get(), frameWidth,
                        b.header.imageNumbers, numberOfPixels,
                        numberOfScanPositions, radialSumHandle);
        // If we don't reset this, it won't get reset until the thread is
        // destroyed.
        b.data.reset();
      }));
#else
    futures.emplace_back(
      pool.enqueue([b, numberOfPixels, centerX, centerY,  frameWidth, frameHeight, &radialSum]() mutable {
        radialSumFrames(centerX, centerY, b.data.get(), frameWidth, frameHeight, b.header.imageNumbers,
            numberOfPixels, radialSum);
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

// Instantiate the ones that can be used
template vector<STEMImage> createSTEMImages<StreamReader::iterator,Block>(StreamReader::iterator first,
                                            StreamReader::iterator last,
                                            vector<int> innerRadii,
                                            vector<int> outerRadii, int width,
                                            int height, int centerX,
                                            int centerY);

template vector<STEMImage> createSTEMImages<H5Reader::iterator,PyBlock>(H5Reader::iterator first,
                                            H5Reader::iterator last,
                                            vector<int> innerRadii,
                                            vector<int> outerRadii, int width,
                                            int height, int centerX,
                                            int centerY);

template vector<STEMImage> createSTEMImages<vector<Block>::iterator,Block>(vector<Block>::iterator first,
                                            vector<Block>::iterator last,
                                            vector<int> innerRadii,
                                            vector<int> outerRadii, int width,
                                            int height, int centerX,
                                            int centerY);

template Image<double> calculateAverage(StreamReader::iterator first,
                                        StreamReader::iterator last);
template Image<double> calculateAverage(vector<Block>::iterator first,
                                        vector<Block>::iterator last);

template RadialSum<uint64_t> radialSum(StreamReader::iterator first, StreamReader::iterator last,
      int scanWidth, int scanHeight, int centerX, int centerY);
template RadialSum<uint64_t> radialSum(vector<Block>::iterator, vector<Block>::iterator last,
      int scanWidth, int scanHeight, int centerX, int centerY);

}
