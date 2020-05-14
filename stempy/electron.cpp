#include "electron.h"
#include "electronthresholds.h"
#include "python/pyreader.h"
#include "reader.h"

#include "config.h"

#ifdef VTKm
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/worklet/WorkletMapTopology.h>
#include <vtkm/worklet/WorkletPointNeighborhood.h>
#endif

#include <sstream>
#include <stdexcept>

using stempy::Dimensions2D;

#ifdef VTKm
namespace {

struct IsMaximalPixel : public vtkm::worklet::WorkletPointNeighborhood
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn,
                                FieldInNeighborhood neighborhood,
                                FieldOut isMaximal);

  using ExecutionSignature = void(_2, _3);

  template <typename NeighIn>
  VTKM_EXEC void operator()(const NeighIn& neighborhood, bool& isMaximal) const
  {
    isMaximal = false;

    auto current = neighborhood.Get(0, 0, 0);
    for (int j = -1; j < 2; ++j) {
      for (int i = -1; i < 2; ++i) {
        if (i == 0 && j == 0)
          continue;
        if (current <= neighborhood.Get(i, j, 0))
          return;
      }
    }

    isMaximal = true;
  }
};

// The types in here, "uint16_t" and "double", are specific for our use case
// We may want to generalize it in the future
struct SubtractAndThreshold : public vtkm::worklet::WorkletMapCellToPoint
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn, FieldInOutPoint value,
                                FieldInPoint background);

  using ExecutionSignature = void(_2, _3);

  VTKM_EXEC void operator()(uint16_t& val, double background) const
  {
    val -= static_cast<uint16_t>(background);
    if (val <= m_lower || val >= m_upper)
      val = 0;
  }

  VTKM_CONT
  SubtractAndThreshold(double lower, double upper)
    : m_lower(lower), m_upper(upper){};

private:
  double m_lower;
  double m_upper;
};

std::vector<uint32_t> maximalPointsParallel(std::vector<uint16_t>& frame,
                                            Dimensions2D frameDimensions,
                                            const double* darkReferenceData,
                                            double backgroundThreshold,
                                            double xRayThreshold)
{
  // Build the data set
  vtkm::cont::CellSetStructured<2> cellSet;
  // frameDimensions.second corresponds to rows, and frameDimensions.first
  // corresponds to columns
  cellSet.SetPointDimensions(
    vtkm::Id2(frameDimensions.second, frameDimensions.first));

  // Input handles
  auto frameHandle = vtkm::cont::make_ArrayHandle(frame);
  auto darkRefHandle = vtkm::cont::make_ArrayHandle(
    darkReferenceData, frameDimensions.first * frameDimensions.second);

  // Output
  vtkm::cont::ArrayHandle<bool> maximalPixels;

  vtkm::cont::Invoker invoke;
  // Background subtraction and thresholding
  invoke(SubtractAndThreshold{ backgroundThreshold, xRayThreshold }, cellSet,
         frameHandle, darkRefHandle);
  // Find maximal pixels
  invoke(IsMaximalPixel{}, cellSet, frameHandle, maximalPixels);

  // Convert to std::vector<uint32_t>
  auto maximalPixelsPortal = maximalPixels.GetPortalConstControl();
  std::vector<uint32_t> outputVec;
  for (vtkm::Id i = 0; i < maximalPixelsPortal.GetNumberOfValues(); ++i) {
    if (maximalPixelsPortal.Get(i))
      outputVec.push_back(i);
  }

  // Done
  return outputVec;
}
} // end namespace
#endif

namespace stempy {

// Implementation of modulus that "wraps" for negative numbers
inline uint16_t mod(uint16_t x, uint16_t y)
{
  return ((x % y) + y) % y;
}

// Return the points in the frame with values larger than all 8 of their nearest
// neighbors
std::vector<uint32_t> maximalPoints(const std::vector<uint16_t>& frame,
                                    Dimensions2D frameDimensions)
{
  auto width = frameDimensions.first;
  auto height = frameDimensions.second;

  std::vector<uint32_t> events;
  auto numberOfPixels = height * width;
  for (uint32_t i = 0; i < numberOfPixels; ++i) {
    auto row = i / width;
    auto column = i % width;
    auto rightNeighbourColumn = mod((i + 1), width);
    auto leftNeighbourColumn = mod((i - 1), width);
    auto topNeighbourRow = mod((row - 1), height);
    auto bottomNeighbourRow = mod((row + 1), height);
    auto pixelValue = frame[i];
    auto bottomNeighbourRowIndex = bottomNeighbourRow * width;
    auto topNeighbourRowIndex = topNeighbourRow * width;
    auto rowIndex = row * width;

    auto event = true;

    // If we are on row 0, there are no pixels above this one
    if (row != 0) {
      // Check top sections
      // top
      event = event && pixelValue > frame[topNeighbourRowIndex + column];
      // top left
      event = event &&
              (column == 0 ||
               pixelValue > frame[topNeighbourRowIndex + leftNeighbourColumn]);
      // top right
      event = event &&
              (column == width - 1 ||
               pixelValue > frame[topNeighbourRowIndex + rightNeighbourColumn]);
    }

    // If we are on the bottom row, there are no pixels below this one
    if (event && row != height - 1) {
      // Check bottom sections
      // bottom
      event = event && pixelValue > frame[bottomNeighbourRowIndex + column];
      // bottom left
      event = event && (column == 0 ||
                        pixelValue >
                          frame[bottomNeighbourRowIndex + leftNeighbourColumn]);
      // bottom right
      event =
        event &&
        (column == width - 1 ||
         pixelValue > frame[bottomNeighbourRowIndex + rightNeighbourColumn]);
    }

    // left
    event = event &&
            (column == 0 || pixelValue > frame[rowIndex + leftNeighbourColumn]);
    // right
    event = event && (column == width - 1 ||
                      pixelValue > frame[rowIndex + rightNeighbourColumn]);

    if (event) {
      events.push_back(i);
    }
  }

  return events;
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const double darkReference[],
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions)
{
  if (first == last) {
    std::ostringstream msg;
    msg << "No blocks to read!";
    throw std::invalid_argument(msg.str());
  }

  // If we haven't been provided with width and height, try the header.
  if (scanDimensions.first == 0 || scanDimensions.second == 0) {
    scanDimensions = first->header.scanDimensions;
  }

  // Raise an exception if we still don't have valid rows and columns
  if (scanDimensions.first <= 0 || scanDimensions.second <= 0) {
    std::ostringstream msg;
    msg << "No scan image size provided.";
    throw std::invalid_argument(msg.str());
  }

  // Store the frameDimensions from the first block
  // It should be the same for all blocks
  auto frameDimensions = first->header.frameDimensions;

  // Matrix to hold electron events.
  std::vector<std::vector<uint32_t>> events(scanDimensions.first *
                                            scanDimensions.second);
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto data = block.data.get();
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto frameStart =
        data + i * frameDimensions.first * frameDimensions.second;
      std::vector<uint16_t> frame(frameStart,
                                  frameStart + frameDimensions.first *
                                                 frameDimensions.second);

#ifdef VTKm
      events[block.header.imageNumbers[i]] =
        maximalPointsParallel(frame, frameDimensions, darkReference,
                              backgroundThreshold, xRayThreshold);
#else
      for (int j = 0; j < frameDimensions.first * frameDimensions.second; j++) {
        // Subtract darkfield reference
        frame[j] -= darkReference[j];
        // Threshold the electron events
        if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
          frame[j] = 0;
        }
      }
      // Now find the maximal events
      events[block.header.imageNumbers[i]] =
        maximalPoints(frame, frameDimensions);
#endif
    }
  }

  ElectronCountedData ret;
  ret.data = events;
  ret.scanDimensions = scanDimensions;
  ret.frameDimensions = frameDimensions;

  return ret;
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<double>& darkReference,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions)
{
  return electronCount(first, last, darkReference.data.get(),
                       backgroundThreshold, xRayThreshold, scanDimensions);
}

std::vector<uint32_t> electronCount(std::vector<uint16_t>& frame,
                                    Dimensions2D frameDimensions,
                                    const double darkReference[],
                                    double backgroundThreshold,
                                    double xRayThreshold)
{

  for (int j = 0; j < frameDimensions.first * frameDimensions.second; j++) {
    // Subtract darkfield reference
    frame[j] -= darkReference[j];
    // Threshold the electron events
    if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
      frame[j] = 0;
    }
  }
  // Now find the maximal events
  return maximalPoints(frame, frameDimensions);
}

ElectronCountedData electronCount(SectorStreamThreadedReader* reader,
                                  Image<double>& darkReference,
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount(
    reader, darkReference.data.get(), thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, scanDimensions, verbose);
}

ElectronCountedData electronCount(SectorStreamThreadedReader* reader,
                                  const double darkReference[],
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  Dimensions2D scanDimensions, bool verbose)
{
  // This is where we will save the electron events as the calculated
  std::vector<std::vector<uint32_t>> events;

  // Used to signal that we have moved into the electron counting phase after
  // collecting samples to calculate threshold.
  std::atomic<bool> electronCounting = { false };

  // These hold the optimized thresholds.
  double backgroundThreshold;
  double xRayThreshold;

  // Mutexes to control/protect counting process
  std::mutex sampleMutex;
  std::condition_variable sampleCondition;

  // The sample blocks that will be used to calculate the thresholds
  std::vector<Block> sampleBlocks;

  auto counter = [&events, &electronCounting, &backgroundThreshold,
                  &xRayThreshold, &sampleMutex, &sampleCondition, &sampleBlocks,
                  thresholdNumberOfBlocks, numberOfSamples,
                  darkReference](Block& b) {
    // If we are still collecting sample block for calculating the threshold
    if (!electronCounting) {
      std::unique_lock<std::mutex> sampleLock(sampleMutex);
      if (sampleBlocks.size() < thresholdNumberOfBlocks) {
        sampleBlocks.push_back(std::move(b));
        if (sampleBlocks.size() == thresholdNumberOfBlocks) {
          sampleLock.unlock();
          sampleCondition.notify_all();
        }
      }
      // We have our samples, so we should wait for the threshold to be
      // calculated in the main thread, before we can start counting.
      else {
        sampleCondition.wait(sampleLock, [&electronCounting]() {
          return electronCounting.load();
        });
      }
    }
    // We are electron counting
    else {
      auto data = b.data.get();
      auto frameDimensions = b.header.frameDimensions;
      for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
        auto frameStart =
          data + i * frameDimensions.first * frameDimensions.second;
        std::vector<uint16_t> frame(frameStart,
                                    frameStart + frameDimensions.first *
                                                   frameDimensions.second);

        auto frameEvents = electronCount(frame, frameDimensions, darkReference,
                                         backgroundThreshold, xRayThreshold);
        events[b.header.imageNumbers[i]] = frameEvents;
      }
    }
  };

  auto done = reader->readAll(counter);

  // Wait for enought blocks to come in to calculate the threadhold.
  std::unique_lock<std::mutex> lock(sampleMutex);
  sampleCondition.wait(lock, [&sampleBlocks, thresholdNumberOfBlocks]() {
    return sampleBlocks.size() == thresholdNumberOfBlocks;
  });

  // Now calculate the threshold
  auto threshold =
    calculateThresholds(sampleBlocks, darkReference, numberOfSamples,
                        backgroundThresholdNSigma, xRayThresholdNSigma);

  if (verbose) {
    std::cout << "****Statistics for calculating electron thresholds****"
              << std::endl;
    std::cout << "number of samples:" << threshold.numberOfSamples << std::endl;
    std::cout << "min sample:" << threshold.minSample << std::endl;
    std::cout << "max sample:" << threshold.maxSample << std::endl;
    std::cout << "mean:" << threshold.mean << std::endl;
    std::cout << "variance:" << threshold.variance << std::endl;
    std::cout << "std dev:" << threshold.stdDev << std::endl;
    std::cout << "number of bins:" << threshold.numberOfBins << std::endl;
    std::cout << "x-ray threshold n sigma:" << threshold.xRayThresholdNSigma
              << std::endl;
    std::cout << "background threshold n sigma:"
              << threshold.backgroundThresholdNSigma << std::endl;
    std::cout << "optimized mean:" << threshold.optimizedMean << std::endl;
    std::cout << "optimized std dev:" << threshold.optimizedStdDev << std::endl;
    std::cout << "background threshold:" << threshold.backgroundThreshold
              << std::endl;
    std::cout << "xray threshold:" << threshold.xRayThreshold << std::endl;
  }

  // Now setup  electron count output
  auto scanSize = sampleBlocks[0].header.scanDimensions;
  auto frameSize = sampleBlocks[0].header.frameDimensions;
  events.resize(scanSize.first * scanSize.second);

  // Now tell our workers to proceed
  electronCounting = true;
  backgroundThreshold = threshold.backgroundThreshold;
  xRayThreshold = threshold.xRayThreshold;
  lock.unlock();
  sampleCondition.notify_all();

  // Count the sample blocks
  for (auto& b : sampleBlocks) {
    auto data = b.data.get();
    auto frameDimensions = b.header.frameDimensions;
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto frameStart =
        data + i * frameDimensions.first * frameDimensions.second;
      std::vector<uint16_t> frame(frameStart,
                                  frameStart + frameDimensions.first *
                                                 frameDimensions.second);
      auto frameEvents = electronCount(frame, frameDimensions, darkReference,
                                       backgroundThreshold, xRayThreshold);
      events[b.header.imageNumbers[i]] = frameEvents;
    }
  }

  // Make sure all threads are finished before returning the result
  done.wait();

  ElectronCountedData ret;
  ret.data = events;
  ret.scanDimensions = scanSize;
  ret.frameDimensions = frameSize;

  return ret;
}

// Instantiate the ones that can be used
template ElectronCountedData electronCount(StreamReader::iterator first,
                                           StreamReader::iterator last,
                                           Image<double>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(SectorStreamReader::iterator first,
                                           SectorStreamReader::iterator last,
                                           Image<double>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(PyReader::iterator first,
                                           PyReader::iterator last,
                                           Image<double>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
}
