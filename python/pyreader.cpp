#include "pyreader.h"
#include <stempy/reader.h>

namespace stempy {

PyBlock::PyBlock(py::array_t<uint16_t> pyarray)
{
  // Since this is the constructor, this should not be deleting anything
  // Otherwise, we might would need py::gil_scoped_acquire gil
  this->data.array.reset(new py::array_t<uint16_t>(std::move(pyarray)));
}

PyReader::PyReader(py::object pyDataSet, std::vector<uint32_t>& imageNumbers,
                   Dimensions2D scanDimensions, uint32_t blockSize,
                   uint32_t totalImageNum)
  : m_pydataset(pyDataSet), m_imageNumbers(imageNumbers),
    m_scanDimensions(scanDimensions), m_imageNumInBlock(blockSize),
    m_totalImageNum(totalImageNum)
{
}

PyBlock PyReader::read()
{
  // Acquire GIL before calling Python code
  py::gil_scoped_acquire acquire;

  // get to the end of the file, return empty Block
  if (m_currIndex >= m_totalImageNum) {
    return PyBlock();
  }

  // get the data pointer from the py object
  auto getItems = m_pydataset.attr("__getitem__");
  uint32_t upperBound = m_currIndex + m_imageNumInBlock;

  if (upperBound > m_totalImageNum) {
    upperBound = m_totalImageNum;
  }

  auto sliceIndex = py::slice(m_currIndex, upperBound, 1);
  py::array_t<uint16_t> pyarray = getItems(sliceIndex);

  PyBlock b(pyarray);
  Dimensions2D frameDimensions = { pyarray.shape()[2], pyarray.shape()[1] };

  // get the image numbers for current header
  std::vector<uint32_t> imageNumberForBlock;
  for (auto i = m_currIndex; i < upperBound; i++) {
    imageNumberForBlock.push_back(m_imageNumbers[i]);
  }

  b.header = Header(frameDimensions, m_imageNumInBlock, m_scanDimensions,
                    imageNumberForBlock);
  b.header.version = 3;

  m_currIndex = upperBound;

  return b;
}

PyReader::iterator PyReader::begin()
{
  // read data at first time
  return iterator(this);
}

PyReader::iterator PyReader::end()
{
  return iterator(nullptr);
}

void PyReader::reset()
{
  m_currIndex = 0;
}

} // namespace stempy
