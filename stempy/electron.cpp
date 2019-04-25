#include "electron.h"
#include "electronthresholds.h"

#include "config.h"

#ifdef VTKm
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/worklet/Invoker.h>
#include <vtkm/worklet/WorkletMapTopology.h>
#include <vtkm/worklet/WorkletPointNeighborhood.h>
#endif

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
    val -= background;
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

std::vector<uint32_t> maximalPointsParallel(
  std::vector<uint16_t>& frame, int rows, int columns,
  double* darkReferenceData, double backgroundThreshold, double xRayThreshold)
{
  // Build the data set
  vtkm::cont::CellSetStructured<2> cellSet("frame");
  cellSet.SetPointDimensions(vtkm::Id2(rows, columns));

  // Input handles
  auto frameHandle = vtkm::cont::make_ArrayHandle(frame);
  auto darkRefHandle =
    vtkm::cont::make_ArrayHandle(darkReferenceData, rows * columns);

  // Output
  vtkm::cont::ArrayHandle<bool> maximalPixels;

  vtkm::worklet::Invoker invoke;
  // Background subtraction and thresholding
  invoke(SubtractAndThreshold{ backgroundThreshold, xRayThreshold }, cellSet,
         frameHandle, darkRefHandle);
  // Find maximal pixels
  invoke(IsMaximalPixel{}, cellSet, frameHandle, maximalPixels);

  // Convert to std::vector<uint32_t>
  auto maximalPixelsPortal = maximalPixels.GetPortalConstControl();
  std::vector<uint32_t> outputVec;
  outputVec.reserve(maximalPixelsPortal.GetNumberOfValues());
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
std::vector<uint32_t> maximalPoints(
  const std::vector<uint16_t>& frame, int rows, int columns)
{
  std::vector<uint32_t> events;
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
      events.push_back(i);
    }
  }

  return events;
}

template <typename InputIt>
std::vector<std::vector<uint32_t>> electronCount(InputIt first, InputIt last,
                                                 int scanRows, int scanColumns,
                                                 Image<double>& darkReference,
                                                 double backgroundThreshold,
                                                 double xRayThreshold)
{
  // Matrix to hold electron events.
  std::vector<std::vector<uint32_t>> events(scanRows * scanColumns);
  for (; first != last; ++first) {
    auto block = std::move(*first);
    auto data = block.data.get();
    for (int i = 0; i < block.header.imagesInBlock; i++) {
      auto frameStart = data + i * block.header.rows * block.header.columns;
      std::vector<uint16_t> frame(
        frameStart, frameStart + block.header.rows * block.header.columns);

#ifdef VTKm
      events[block.header.imageNumbers[i]] = maximalPointsParallel(
        frame, block.header.rows, block.header.columns,
        darkReference.data.get(), backgroundThreshold, xRayThreshold);
#else
      for (int j = 0; j < block.header.rows * block.header.columns; j++) {
        // Subtract darkfield reference
        frame[j] -= darkReference.data[j];
        // Threshold the electron events
        if (frame[j] <= backgroundThreshold || frame[j] >= xRayThreshold) {
          frame[j] = 0;
        }
      }
      // Now find the maximal events
      events[block.header.imageNumbers[i]] =
        maximalPoints(frame, block.header.rows, block.header.columns);
#endif
    }
  }

  return events;
}

// Instantiate the ones that can be used
template std::vector<std::vector<uint32_t>> electronCount(
  StreamReader::iterator first, StreamReader::iterator last, int scanRows,
  int scanColumns, Image<double>& darkReference, double backgroundThreshold,
  double xRayThreshold);
}
