#include "electron.h"

#include <algorithm>
#include <cmath>
#include <random>

#include <lsq/lsqcpp.h>

#include "config.h"

#ifdef VTKm
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/filter/FilterDataSet.h>
#include <vtkm/worklet/DispatcherPointNeighborhood.h>
#include <vtkm/worklet/WorkletPointNeighborhood.h>
#endif

#ifdef VTKm
namespace {

struct GetMaximalPixelsPolicy : public vtkm::filter::PolicyBase<GetMaximalPixelsPolicy>
{
  using FieldTypeList = vtkm::ListTagBase<vtkm::UInt8, vtkm::UInt16>;
};

struct IsMaximalPixel : public vtkm::worklet::WorkletPointNeighborhood
{
  using CountingHandle = vtkm::cont::ArrayHandleCounting<vtkm::Id>;

  using ControlSignature = void(CellSetIn,
                                FieldInNeighborhood neighborhood,
                                FieldOut isMaximal);

  using ExecutionSignature = void(_2, _3);

  template <typename NeighIn>
  VTKM_EXEC void operator()(const NeighIn& neighborhood,
                            vtkm::UInt8& isMaximal) const
  {
    isMaximal = 0;

    auto current = neighborhood.Get(0, 0, 0);
    for (int i = -1; i < 2; ++i) {
      for (int j = -1; j < 2; ++j) {
        if (i == 0 && j == 0)
          continue;
        if (current <= neighborhood.Get(i, j, 0))
          return;
      }
    }

    isMaximal = 1;
  }
};

class GetMaximalPixels : public vtkm::filter::FilterDataSet<GetMaximalPixels>
{
public:
  template <typename Policy>
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input,
                                          vtkm::filter::PolicyBase<Policy> policy)

  {
    using DispatcherType = vtkm::worklet::DispatcherPointNeighborhood<IsMaximalPixel>;

    vtkm::cont::ArrayHandle<vtkm::UInt16> inputPixels;
    vtkm::cont::ArrayHandle<vtkm::UInt8> maximalPixels;

    // Get the coordinate system we are using for the 2D area
    const vtkm::cont::DynamicCellSet& cells = input.GetCellSet(this->GetActiveCellSetIndex());

    // Get the input pixels
    input.GetField("inputPixels", vtkm::cont::Field::Association::POINTS).GetData().CopyTo(inputPixels);

    // Execute
    DispatcherType dispatcher;
    dispatcher.Invoke(vtkm::filter::ApplyPolicy(cells, policy), inputPixels, maximalPixels);

    // Get the output
    vtkm::cont::DataSet output;
    output.AddCellSet(cells);
    output.AddCoordinateSystem(input.GetCoordinateSystem(this->GetActiveCoordinateSystemIndex()));

    vtkm::cont::Field maximalPixelsField("maximalPixels", vtkm::cont::Field::Association::POINTS, maximalPixels);
    output.AddField(maximalPixelsField);

    return output;
  }

  template <typename T, typename StorageType, typename DerivedPolicy>
  VTKM_CONT bool DoMapField(vtkm::cont::DataSet&,
                            const vtkm::cont::ArrayHandle<T, StorageType>&,
                            const vtkm::filter::FieldMetadata&,
                            vtkm::filter::PolicyBase<DerivedPolicy>)
  {
    return false;
  }
};

std::vector<std::pair<int, int>> maximalPointsParallel(
  const std::vector<uint16_t>& frame, int rows, int columns)
{
  // Build the data set
  vtkm::cont::DataSetBuilderUniform builder;
  vtkm::cont::DataSet data = builder.Create(vtkm::Id2(rows, columns));

  // Set the input pixels
  auto inputPixelsField =
    vtkm::cont::make_Field("inputPixels", vtkm::cont::Field::Association::POINTS, frame);
  data.AddField(inputPixelsField);

  // Execute
  GetMaximalPixels filter;
  vtkm::cont::DataSet result = filter.Execute(data, GetMaximalPixelsPolicy{});

  // Get the output
  vtkm::cont::ArrayHandle<vtkm::UInt8> maximalPixels;
  result.GetField("maximalPixels", vtkm::cont::Field::Association::POINTS).GetData().CopyTo(maximalPixels);

  // Convert to std::vector<std::pair<int, int>>
  std::vector<std::pair<int, int>> outputVec;
  auto maximalPixelsPortal = maximalPixels.GetPortalConstControl();
  for (vtkm::Id i = 0; i < maximalPixelsPortal.GetNumberOfValues(); ++i) {
    if (maximalPixelsPortal.Get(i) == 1) {
      auto row = i / columns;
      auto column = i % columns;
      outputVec.push_back(std::make_pair(row, column));
    }
  }

  // Done
  return outputVec;
}
} // end namespace
#endif

namespace stempy {

double calculateMean(std::vector<int16_t>& values)
{
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double calculateVariance(std::vector<int16_t>& values, double mean)
{
  double variance = 0;

  for (int i = 0; i < values.size(); i++) {
    variance += pow(values[i], 2.0);
  }

  return (variance / (values.size() - 1)) - pow(mean, 2.0);
}

class GaussianErrorFunction : public lsq::ErrorFunction<double>
{

public:
  GaussianErrorFunction(const std::vector<int32_t>& bins,
                        const std::vector<uint64_t>& histogram)
    : bins(bins), histogram(histogram)
  {
  }
  void _evaluate(const lsq::Vectord& state, lsq::Vectord& outVal,
                 lsq::Matrixd&) override
  {
    outVal.resize(bins.size());
    for (Eigen::Index i = 0; i < bins.size(); ++i) {
      outVal[i] =
        (state[0] * exp(-0.5 * pow((this->bins[i] - state[1]) / state[2], 2)) -
         histogram[i]);
    }
  }

private:
  const std::vector<int32_t>& bins;
  const std::vector<uint64_t>& histogram;
};

std::pair<double, double> calculateThresholds(std::vector<Block>& blocks,
                                              Image<double>& darkReference,
                                              int numberOfSamples,
                                              int backgroundThresholdNSigma,
                                              int xRayThresholdNSigma)
{
  auto detectorImageRows = blocks[0].header.rows;
  auto detectorImageColumns = blocks[0].header.columns;
  auto numberOfPixels = detectorImageRows * detectorImageRows;

  // Setup random number engine
  std::random_device randomDevice;
  std::default_random_engine randomEngine(randomDevice());

  int numberSamplePixels =
    detectorImageRows * detectorImageColumns * numberOfSamples;
  std::vector<int16_t> samples(numberSamplePixels, 0);
  for (int i = 0; i < numberOfSamples; i++) {
    std::uniform_int_distribution<int> randomBlockDist(0, blocks.size() - 1);
    auto randomBlockIndex = randomBlockDist(randomEngine);
    auto randomBlock = blocks[randomBlockIndex];
    std::uniform_int_distribution<int> randomFrameDist(
      0, randomBlock.header.imagesInBlock - 1);
    auto randomFrameIndex = randomFrameDist(randomEngine);
    auto blockData = randomBlock.data.get();

    for (int j = 0; j < numberOfPixels; j++) {
      // For now just use the index, the image number don't seem to work, in the
      // current data set. In the future we should be using the image number.
      samples[i * numberOfPixels + j] =
        blockData[randomFrameIndex * numberOfPixels + j] -
        static_cast<int16_t>(darkReference.data[j]);
    }
  }

  // Calculate stats
  auto mean = calculateMean(samples);
  auto variance = calculateVariance(samples, mean);
  auto stdDev = sqrt(variance);
  auto xrayThreshold = mean + xRayThresholdNSigma * stdDev;

  // Now generate a histograms
  auto minMax = std::minmax_element(samples.begin(), samples.end());
  auto minSample = *minMax.first;
  auto maxSample = std::ceil(*minMax.second);
  auto maxBin = std::min(static_cast<int>(maxSample),
                         static_cast<int>(mean + xrayThreshold * stdDev));
  auto minBin = std::max(static_cast<int>(minSample),
                         static_cast<int>(mean - xrayThreshold * stdDev));

  auto numberOfBins = maxBin - minBin;
  std::vector<uint64_t> histogram(numberOfBins, 0);
  std::vector<int32_t> bins(numberOfBins);

  auto binEdge = minBin;
  for (int i = 0; i < numberOfBins; i++) {
    bins[i] = binEdge++;
  }

  // Bin the values
  for (int i = 0; i < numberSamplePixels; i++) {
    auto binIndex = static_cast<int>(samples[i] - minBin);
    // Skip values outside range
    if (binIndex >= numberOfBins) {
      continue;
    }
    histogram[binIndex] += 1;
  }

  // Now optimize to file Gaussian
  lsq::LevenbergMarquardt<double> optAlgo;
  optAlgo.setLineSearchAlgorithm(nullptr);
  optAlgo.setMaxIterationsLM(20);
  optAlgo.setEpsilon(1e-6);
  GaussianErrorFunction* errorFunction =
    new GaussianErrorFunction(bins, histogram);
  optAlgo.setErrorFunction(errorFunction);
  lsq::Vectord initialState(3);
  auto indexOfMaxElement =
    std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
  initialState[0] = static_cast<double>(histogram[indexOfMaxElement]);
  initialState[1] =
    (bins[indexOfMaxElement + 1] - bins[indexOfMaxElement]) / 2.0;
  initialState[2] = stdDev;

  // optimize
  auto result = optAlgo.optimize(initialState);

  if (!result.converged) {
    throw std::runtime_error("Optimization did not converge");
  }
  auto backgroundThreshold =
    result.state[1] + result.state[2] * backgroundThresholdNSigma;

  return std::make_pair(backgroundThreshold, xrayThreshold);
}

// Implementation of modulus that "wraps" for negative numbers
inline uint16_t mod(uint16_t x, uint16_t y)
{
  return ((x % y) + y) % y;
}

// Return the points in the frame with values larger than all 8 of their nearest
// neighbors
std::vector<std::pair<int, int>> maximalPoints(
  const std::vector<uint16_t>& frame, int rows, int columns)
{
  std::vector<std::pair<int, int>> events;
  auto numberOfPixels = rows * columns;
  for (int i = 0; i < numberOfPixels; i++) {
    auto row = i / columns;
    auto column = i % columns;
    auto rightNeighbourColumn = mod((i + 1), columns);
    auto leftNeighbourColumn = mod((i - 1), columns);
    auto topNeighbourRow = mod((row - 1), rows);
    auto bottomNeighbourRow = mod((row + 1), rows);
    auto pixelValue = frame[i];
    auto bottomNeighbourRowIndex = bottomNeighbourRow * columns;
    auto topNeighbourRowIndex = topNeighbourRow * columns;
    auto rowIndex = row * columns;

    // top
    auto event = pixelValue > frame[topNeighbourRowIndex + column];
    // top right
    event =
      event && pixelValue > frame[topNeighbourRowIndex + rightNeighbourColumn];
    // right
    event = event && pixelValue > frame[rowIndex + rightNeighbourColumn];
    // bottom right
    event = event &&
            pixelValue > frame[bottomNeighbourRowIndex + rightNeighbourColumn];
    // bottom
    event = event && pixelValue > frame[bottomNeighbourRowIndex + column];
    // bottom left
    event = event &&
            pixelValue > frame[bottomNeighbourRowIndex + leftNeighbourColumn];
    // left
    event = event && pixelValue > frame[rowIndex + leftNeighbourColumn];
    // top left
    event =
      event && pixelValue > frame[topNeighbourRowIndex + leftNeighbourColumn];

    if (event) {
      events.push_back(std::make_pair(row, column));
    }
  }

  return events;
}

std::vector<std::vector<std::pair<int, int>>> electronCount(
  std::vector<Block>& blocks, int scanRows, int scanColumns,
  Image<double>& darkReference, int numberOfSamples,
  int backgroundThresholdNSigma, int xRayThresholdNSigma)
{
  std::pair<double, double> thres =
    calculateThresholds(blocks, darkReference, numberOfSamples,
                        backgroundThresholdNSigma, xRayThresholdNSigma);
  auto backgroundThreshold = std::get<0>(thres);
  auto xRayThreshold = std::get<1>(thres);

  // Matrix to hold electron events.
  std::vector<std::vector<std::pair<int, int>>> events(scanRows * scanColumns);
  int frameIndex = 0;
  for (const Block& block : blocks) {
    auto data = block.data.get();
    for (int i = 0; i < block.header.imagesInBlock; i++) {
      auto frameStart = data + i * block.header.rows * block.header.columns;
      std::vector<uint16_t> frame(
        frameStart, frameStart + block.header.rows * block.header.columns);
      for (int j = 0; j < block.header.rows * block.header.columns; j++) {
        // Subtract darkfield reference
        frame[j] -= darkReference.data[j];
        // Threshold the electron events
        if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
          frame[j] = 0;
        }
      }

#ifdef VTKm
      events[block.header.imageNumbers[i]] =
        maximalPointsParallel(frame, block.header.rows, block.header.columns);
#else
      // Now find the maximal events
      events[block.header.imageNumbers[i]] =
        maximalPoints(frame, block.header.rows, block.header.columns);
#endif
    }
  }

  return events;
}
}
