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
struct Threshold : public vtkm::worklet::WorkletMapCellToPoint
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn, FieldInOutPoint value);

  using ExecutionSignature = void(_2);

  template <typename FrameType>
  VTKM_EXEC void operator()(FrameType& val) const
  {
    if (val <= m_lower || val >= m_upper)
      val = 0;
  }

  VTKM_CONT
  Threshold(double lower, double upper) : m_lower(lower), m_upper(upper){};

private:
  double m_lower;
  double m_upper;
};

struct SubtractAndThreshold : public Threshold
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn, FieldInOutPoint value,
                                FieldInPoint background);

  using ExecutionSignature = void(_2, _3);

  template <typename FrameType>
  VTKM_EXEC void operator()(FrameType& val, double background) const
  {
    val -= static_cast<FrameType>(background);

    Threshold::operator()(val);
  }

  VTKM_CONT
  SubtractAndThreshold(double lower, double upper) : Threshold(lower, upper){};

private:
  double m_lower;
  double m_upper;
};

struct ApplySubtractGainAndThreshold : public Threshold
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn, FieldInOutPoint value,
                                FieldInPoint background, FieldInPoint gain);

  using ExecutionSignature = void(_2, _3, _4);

  template <typename FrameType>
  VTKM_EXEC void operator()(FrameType& val, double background, float gain) const
  {

    val = static_cast<FrameType>((val - static_cast<float>(background)) * gain);

    Threshold::operator()(val);
  }

  VTKM_CONT
  ApplySubtractGainAndThreshold(double lower, double upper)
    : Threshold(lower, upper){};

private:
  double m_lower;
  double m_upper;
};

struct ApplyGainAndThreshold : public Threshold
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn, FieldInOutPoint value,
                                FieldInPoint background);

  using ExecutionSignature = void(_2, _3);

  template <typename FrameType>
  VTKM_EXEC void operator()(FrameType& val, float gain) const
  {

    val = static_cast<FrameType>(val * gain);

    Threshold::operator()(val);
  }

  VTKM_CONT
  ApplyGainAndThreshold(double lower, double upper) : Threshold(lower, upper){};

private:
  double m_lower;
  double m_upper;
};

template <typename FrameType, bool dark = true>
std::vector<uint32_t> maximalPointsParallel(std::vector<FrameType>& frame,
                                            Dimensions2D frameDimensions,
                                            const float* darkReferenceData,
                                            const float* gain,
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

  // Output
  vtkm::cont::ArrayHandle<bool> maximalPixels;

  vtkm::cont::Invoker invoke;

  // Call the correct worklet based on whether we are applying a gain, which in
  // term determines the type, so can be evaluated a compile time.
  // First no gain
  if (std::is_integral<FrameType>::value) {
    // Background subtraction and thresholding

    // static if to determine if we are going to subtract dark reference
    static_if<dark>(
      [&]() {
        auto darkRefHandle = vtkm::cont::make_ArrayHandle(
          darkReferenceData, frameDimensions.first * frameDimensions.second);

        invoke(SubtractAndThreshold{ backgroundThreshold, xRayThreshold },
               cellSet, frameHandle, darkRefHandle);
      },
      [&] {
        invoke(Threshold{ backgroundThreshold, xRayThreshold }, cellSet,
               frameHandle);
      })();
  }
  // We are applying a gain
  else {
    auto gainRefHandle = vtkm::cont::make_ArrayHandle(
      gain, frameDimensions.first * frameDimensions.second);
    // Apply gain, background subtraction and thresholding
    // static if to determine if we are going to subtract dark reference
    static_if<dark>(
      [&]() {
        auto darkRefHandle = vtkm::cont::make_ArrayHandle(
          darkReferenceData, frameDimensions.first * frameDimensions.second);

        invoke(
          ApplySubtractGainAndThreshold{ backgroundThreshold, xRayThreshold },
          cellSet, frameHandle, darkRefHandle, gainRefHandle);
      },
      [&] {
        invoke(ApplyGainAndThreshold{ backgroundThreshold, xRayThreshold },
               cellSet, frameHandle, gainRefHandle);
      })();
  }
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
template <typename FrameType>
std::vector<uint32_t> maximalPoints(const std::vector<FrameType>& frame,
                                    Dimensions2D frameDimensions)
{
  auto width = frameDimensions.first;
  auto height = frameDimensions.second;

  std::vector<uint32_t> events;
  auto numberOfPixels = height * width;
  for (uint32_t i = 0; i < numberOfPixels; ++i) {
    auto row = i / width;
    auto column = i % width;
    auto rightNeighbourColumn = mod((column + 1), width);
    auto leftNeighbourColumn = mod((column - 1), width);
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

template <typename InputIt, typename FrameType, bool dark = true>
ElectronCountedData electronCount(
  InputIt first, InputIt last, const float darkReference[], const float gain[],
  double backgroundThreshold, double xRayThreshold, Dimensions2D scanDimensions)
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
  // Assume only one frame per scan.
  Events events(scanDimensions.first * scanDimensions.second);
  for (size_t i = 0; i < events.size(); ++i) {
    events[i].resize(1);
  }

  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto data = block.data.get();
    for (unsigned i = 0; i < block.header.imagesInBlock; i++) {
      auto frameStart =
        data + i * frameDimensions.first * frameDimensions.second;
      std::vector<FrameType> frame(frameStart,
                                  frameStart + frameDimensions.first *
                                                 frameDimensions.second);

#ifdef VTKm
      events[block.header.imageNumbers[i]][0] =
        maximalPointsParallel<FrameType, dark>(
          frame, frameDimensions, darkReference, gain, backgroundThreshold,
          xRayThreshold);
#else
      for (int j = 0; j < frameDimensions.first * frameDimensions.second; j++) {
        // Subtract darkfield reference and apply gain if we have one, this will
        // be based on our template type, it can be evaluated a compile time.
        if (std::is_integral<FrameType>::value) {
          frame[j] -= darkReference[j];
        } else {
          frame[j] = (frame[j] - darkReference[j]) * gain[j];
        }
        // Threshold the electron events
        if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
          frame[j] = 0;
        }
      }
      // Now find the maximal events
      events[block.header.imageNumbers[i]][0] =
        maximalPoints<FrameType>(frame, frameDimensions);
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
                                  const float darkReference[],
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, uint16_t>(first, last, darkReference, nullptr,
                                          backgroundThreshold, xRayThreshold,
                                          scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  const float darkReference[],
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, float>(first, last, darkReference, gain,
                                       backgroundThreshold, xRayThreshold,
                                       scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<float>& darkReference,
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, float>(first, last, darkReference.data.get(),
                                       gain, backgroundThreshold, xRayThreshold,
                                       scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  Image<float>& darkReference,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, uint16_t>(first, last, darkReference.data.get(),
                                          nullptr, backgroundThreshold,
                                          xRayThreshold, scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  double backgroundThreshold,
                                  double xRayThreshold,
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, uint16_t, false>(first, last, nullptr, nullptr,
                                                 backgroundThreshold,
                                                 xRayThreshold, scanDimensions);
}

template <typename InputIt>
ElectronCountedData electronCount(InputIt first, InputIt last,
                                  double backgroundThreshold,
                                  double xRayThreshold, const float gain[],
                                  Dimensions2D scanDimensions)
{
  return electronCount<InputIt, float, false>(first, last, nullptr, gain,
                                              backgroundThreshold,
                                              xRayThreshold, scanDimensions);
}

template <typename FrameType, bool applyGlobalDark = true, bool applyGain = true>
void applyRowDark(std::vector<FrameType>& frame, Dimensions2D frameDimensions,
                  const float globalDark[], const float gain[])
{
  auto height = frameDimensions.first;
  auto width = frameDimensions.second;
  auto numberOfPixels = height * width;

  auto leftSize = width / 2;
  auto rightSize = width - leftSize;

  auto mean = [&frame](size_t start, size_t stop)
  {
    return std::accumulate(frame.begin() + start, frame.begin() + stop, 0.0) / static_cast<double>(stop - start);
  };

  // We can choose here between mean and median
  decltype(mean) statFunc = mean;

  double optimized_mean = 20;
  auto apply = [&](size_t start, size_t stop)
  {
    auto statResult = statFunc(start, stop);

    for (size_t i = start; i < stop; ++i) {
      // Apply global dark if needed
      static_if<applyGlobalDark>(
        [&]() { frame[i] -= globalDark[i]; }
      )();

      // Apply row dark algorithm
      frame[i] = optimized_mean * frame[i] / statResult;

      // Apply gain if needed
      static_if<applyGlobalDark>(
        [&]() { frame[i] *= gain[i]; }
      )();
    }
  };

  // Loop over the rows one at a time, applying the algorithm to the left
  // and right sides of the row.
  for (uint32_t row = 0; row < height; ++row) {
    auto i = row * width;
    apply(i, i + leftSize);
    apply(i + leftSize, i + leftSize + rightSize);
  }
}

template <typename FrameType, bool dark = true>
std::vector<uint32_t> electronCount(std::vector<FrameType>& frame,
                                    Dimensions2D frameDimensions,
                                    const float darkReference[],
                                    double backgroundThreshold,
                                    double xRayThreshold, const float gain[])
{

  for (unsigned j = 0; j < frameDimensions.first * frameDimensions.second;
       j++) {
    // Subtract darkfield reference and apply gain if we have one, this will
    // be based on our template type, it can be evaluated a compile time.
    static_if<dark>(
      [&]() { frame[j] -= static_cast<FrameType>(darkReference[j]); })();

    if (not std::is_integral<FrameType>::value) {
      frame[j] *= gain[j];
    }

    // Threshold the electron events
    if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
      frame[j] = 0;
    }
  }
  // Now find the maximal events
  return maximalPoints<FrameType>(frame, frameDimensions);
}

std::vector<uint32_t> electronCount(std::vector<uint16_t>& frame,
                                    Dimensions2D frameDimensions,
                                    const float darkReference[],
                                    double backgroundThreshold,
                                    double xRayThreshold)
{
  return electronCount<uint16_t>(frame, frameDimensions, darkReference,
                                 backgroundThreshold, xRayThreshold, nullptr);
}

std::vector<uint32_t> electronCount(std::vector<float>& frame,
                                    Dimensions2D frameDimensions,
                                    const float darkReference[],
                                    double backgroundThreshold,
                                    double xRayThreshold, const float gain[])
{
  return electronCount<float>(frame, frameDimensions, darkReference,
                              backgroundThreshold, xRayThreshold, gain);
}

template <typename Reader, typename FrameType, bool dark = true>
ElectronCountedData electronCount(Reader* reader, const float darkReference[],
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  const float gain[],
                                  Dimensions2D scanDimensions, bool verbose)
{
  // This is where we will save the electron events as the calculated
  // Outer vector is scan position, middle vector is a frame at that
  // scan position, and inner vector is the electron counted data for
  // the frame.
  Events events;
  // We need an array of mutexes to protect updates for each event vector for a
  // given position, as we may have muliple frames per location.
  std::unique_ptr<std::mutex[]> positionMutexes;

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

#ifdef USE_MPI
  int rank, worldSize;
  initMpiWorldRank(worldSize, rank);

  // Calculate sample to take from each node
  thresholdNumberOfBlocks =
    getSampleBlocksPerRank(worldSize, rank, thresholdNumberOfBlocks);
#endif

  auto countBlock = [&events, &electronCounting, &backgroundThreshold,
                     &xRayThreshold, darkReference, gain,
                     &positionMutexes](Block& b) {
    auto data = b.data.get();
    auto frameDimensions = b.header.frameDimensions;
    for (unsigned i = 0; i < b.header.imagesInBlock; i++) {
      auto frameStart =
        data + i * frameDimensions.first * frameDimensions.second;
      std::vector<FrameType> frame(frameStart,
                                   frameStart + frameDimensions.first *
                                                  frameDimensions.second);

      auto frameEvents = electronCount<FrameType, dark>(
        frame, frameDimensions, darkReference, backgroundThreshold,
        xRayThreshold, gain);

      auto position = b.header.imageNumbers[i];
      auto& eventsForPosition = events[position];
      // Find the mutex for this position and lock it
      auto& mutex = positionMutexes[position];
      std::unique_lock<std::mutex> positionLock(mutex);
      // Append the events
      eventsForPosition.push_back(std::move(frameEvents));
    }
  };

  auto counter = [&events, &electronCounting, &sampleMutex, &sampleCondition,
                  &sampleBlocks, thresholdNumberOfBlocks,
                  &countBlock](Block& b) {
    // If we are still collecting sample block for calculating the threshold
    if (!electronCounting) {
      std::unique_lock<std::mutex> sampleLock(sampleMutex);
      if (sampleBlocks.size() <
          static_cast<unsigned>(thresholdNumberOfBlocks)) {
        sampleBlocks.push_back(std::move(b));
        if (sampleBlocks.size() ==
            static_cast<unsigned>(thresholdNumberOfBlocks)) {
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
        // Make sure we count the block.
        countBlock(b);
      }
    }
    // We are electron counting
    else {
      countBlock(b);
    }
  };

  auto done = reader->readAll(counter);

  // Wait for enough blocks to come in to calculate the threshold.
  std::unique_lock<std::mutex> lock(sampleMutex);
  sampleCondition.wait(lock, [&sampleBlocks, thresholdNumberOfBlocks]() {
    return sampleBlocks.size() ==
           static_cast<unsigned>(thresholdNumberOfBlocks);
  });

  auto calculateThreshold = true;

#ifdef USE_MPI
  gatherBlocks(worldSize, rank, sampleBlocks);
  // Only calculate threshold on rank 0
  calculateThreshold = rank == 0;
#endif

  CalculateThresholdsResults<FrameType> threshold;

  if (calculateThreshold) {
    // Now calculate the threshold
    threshold = calculateThresholds<Block, FrameType, dark>(
      sampleBlocks, darkReference, numberOfSamples, backgroundThresholdNSigma,
      xRayThresholdNSigma, gain);

    if (verbose) {
      std::cout << "****Statistics for calculating electron thresholds****"
                << std::endl;
      std::cout << "number of samples:" << threshold.numberOfSamples
                << std::endl;
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
      std::cout << "optimized std dev:" << threshold.optimizedStdDev
                << std::endl;
      std::cout << "background threshold:" << threshold.backgroundThreshold
                << std::endl;
      std::cout << "xray threshold:" << threshold.xRayThreshold << std::endl;
    }
  }

  backgroundThreshold = threshold.backgroundThreshold;
  xRayThreshold = threshold.xRayThreshold;

#ifdef USE_MPI
  broadcastThresholds(backgroundThreshold, xRayThreshold);
#endif

  // Now setup  electron count output
  auto frameSize = sampleBlocks[0].header.frameDimensions;

  // If we haven't been provided with width and height, try the header.
  if (scanDimensions.first == 0 || scanDimensions.second == 0) {
    scanDimensions = sampleBlocks[0].header.scanDimensions;
  }

  auto numberOfScanPositions = scanDimensions.first * scanDimensions.second;
  events.resize(numberOfScanPositions);
  // Allocate mutexes to protect the event vector at each scan position
  positionMutexes.reset(new std::mutex[numberOfScanPositions]);

  // Now tell our workers to proceed
  electronCounting = true;
  lock.unlock();
  sampleCondition.notify_all();

  // Count the sample blocks
  for (auto i = 0; i < thresholdNumberOfBlocks; i++) {
    auto& b = sampleBlocks[i];
    auto data = b.data.get();
    auto frameDimensions = b.header.frameDimensions;
    for (unsigned j = 0; j < b.header.imagesInBlock; j++) {
      auto frameStart =
        data + j * frameDimensions.first * frameDimensions.second;
      std::vector<FrameType> frame(frameStart,
                                   frameStart + frameDimensions.first *
                                                  frameDimensions.second);
      auto frameEvents = electronCount<FrameType, dark>(
        frame, frameDimensions, darkReference, backgroundThreshold,
        xRayThreshold, gain);
      auto position = b.header.imageNumbers[j];
      events[position].push_back(std::move(frameEvents));
    }
  }

  // Make sure all threads are finished before returning the result
  done.wait();

#ifdef USE_MPI
  gatherEvents(worldSize, rank, events);
#endif

  // Find the maximum number of frames in a position, and make sure all
  // scan positions have this number of frames.
  int maxNumFrames = 0;
  for (int i = 0; i < events.size(); ++i) {
    if (events[i].size() > maxNumFrames) {
      maxNumFrames = events[i].size();
    }
  }

  // Now make sure they are all the same size
  for (int i = 0; i < events.size(); ++i) {
    events[i].resize(maxNumFrames);
  }

  ElectronCountedData ret;
  ret.data = std::move(events);
  ret.scanDimensions = scanDimensions;
  ret.frameDimensions = frameSize;

  auto& metadata = ret.metadata;

  metadata.thresholdCalculated = calculateThreshold;
  metadata.backgroundThreshold = threshold.backgroundThreshold;
  metadata.xRayThreshold = threshold.xRayThreshold;
  metadata.numberOfSamples = threshold.numberOfSamples;
  metadata.minSample = threshold.minSample;
  metadata.maxSample = threshold.maxSample;
  metadata.mean = threshold.mean;
  metadata.variance = threshold.variance;
  metadata.stdDev = threshold.stdDev;
  metadata.numberOfBins = threshold.numberOfBins;
  metadata.xRayThresholdNSigma = threshold.xRayThresholdNSigma;
  metadata.backgroundThresholdNSigma = threshold.backgroundThresholdNSigma;
  metadata.optimizedMean = threshold.optimizedMean;
  metadata.optimizedStdDev = threshold.optimizedStdDev;

  return ret;
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, const float darkReference[],
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  const float gain[],
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, float>(
    reader, darkReference, thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, gain, scanDimensions,
    verbose);
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, Image<float>& darkReference,
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  const float gain[],
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, float>(
    reader, darkReference.data.get(), thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, gain, scanDimensions,
    verbose);
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, const float darkReference[],
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, uint16_t>(
    reader, darkReference, thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, nullptr, scanDimensions,
    verbose);
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, Image<float>& darkReference,
                                  int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, uint16_t>(
    reader, darkReference.data.get(), thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, nullptr, scanDimensions,
    verbose);
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  const float gain[],
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, float, false>(
    reader, nullptr, thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, gain, scanDimensions,
    verbose);
}

template <typename Reader>
ElectronCountedData electronCount(Reader* reader, int thresholdNumberOfBlocks,
                                  int numberOfSamples,
                                  double backgroundThresholdNSigma,
                                  double xRayThresholdNSigma,
                                  Dimensions2D scanDimensions, bool verbose)
{
  return electronCount<Reader, uint16_t, false>(
    reader, nullptr, thresholdNumberOfBlocks, numberOfSamples,
    backgroundThresholdNSigma, xRayThresholdNSigma, nullptr, scanDimensions,
    verbose);
}

// Instantiate the ones that can be used

// With gain and dark reference
template ElectronCountedData electronCount(
  StreamReader::iterator first, StreamReader::iterator last,
  Image<float>& darkReference, double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  StreamReader::iterator first, StreamReader::iterator last,
  const float darkReference[], double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  SectorStreamReader::iterator first, SectorStreamReader::iterator last,
  Image<float>& darkReference, double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  SectorStreamReader::iterator first, SectorStreamReader::iterator last,
  const float darkReference[], double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  PyReader::iterator first, PyReader::iterator last,
  Image<float>& darkReference, double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  PyReader::iterator first, PyReader::iterator last,
  const float darkReference[], double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);

// Without gain and with dark reference
template ElectronCountedData electronCount(StreamReader::iterator first,
                                           StreamReader::iterator last,
                                           Image<float>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(StreamReader::iterator first,
                                           StreamReader::iterator last,
                                           const float darkReference[],
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(SectorStreamReader::iterator first,
                                           SectorStreamReader::iterator last,
                                           Image<float>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(SectorStreamReader::iterator first,
                                           SectorStreamReader::iterator last,
                                           const float darkReference[],
                                           const double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(PyReader::iterator first,
                                           PyReader::iterator last,
                                           Image<float>& darkReference,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(PyReader::iterator first,
                                           PyReader::iterator last,
                                           const float darkReference[],
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);

// With gain and without dark reference
template ElectronCountedData electronCount(StreamReader::iterator first,
                                           StreamReader::iterator last,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           const float gain[],
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(SectorStreamReader::iterator first,
                                           SectorStreamReader::iterator last,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           const float gain[],
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(
  PyReader::iterator first, PyReader::iterator last, double backgroundThreshold,
  double xRayThreshold, const float gain[], Dimensions2D scanDimensions);

// Without gain and without dark reference
template ElectronCountedData electronCount(StreamReader::iterator first,
                                           StreamReader::iterator last,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(SectorStreamReader::iterator first,
                                           SectorStreamReader::iterator last,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);
template ElectronCountedData electronCount(PyReader::iterator first,
                                           PyReader::iterator last,
                                           double backgroundThreshold,
                                           double xRayThreshold,
                                           Dimensions2D scanDimensions);

// Instantiate for threaded readers

// SectorStreamThreadedReader
template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, Image<float>& darkreference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, const float darkreference[],
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, Image<float>& darkreference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[], Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, const float darkreference[],
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[], Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[], Dimensions2D scanDimensions,
  bool verbose);

template ElectronCountedData electronCount(
  SectorStreamThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, Dimensions2D scanDimensions, bool verbose);

// SectorStreamMultiPassThreadedReader
template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, Image<float>& darkreference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, const float darkreference[],
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, Image<float>& darkreference,
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[], Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, const float darkreference[],
  int thresholdNumberOfBlocks, int numberOfSamples,
  double backgroundThresholdNSigma, double xRayThresholdNSigma,
  const float gain[], Dimensions2D scanDimensions, bool verbose);

template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, const float gain[], Dimensions2D scanDimensions,
  bool verbose);

template ElectronCountedData electronCount(
  SectorStreamMultiPassThreadedReader* reader, int thresholdNumberOfBlocks,
  int numberOfSamples, double backgroundThresholdNSigma,
  double xRayThresholdNSigma, Dimensions2D scanDimensions, bool verbose);
}
