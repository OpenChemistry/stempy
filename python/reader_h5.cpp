#include "reader_h5.h"
#include <stempy/reader.h>

namespace stempy {

H5Reader::H5Reader(py::object h5DataSet,std::vector<uint32_t>& imageNumbers, uint32_t imageWidth,
                   uint32_t imageHeight, uint32_t scanWidth,
                   uint32_t scanHeight, uint32_t blockSize,
                   uint32_t blockNumInFile, uint32_t totalImageNum)
  : m_h5dataset(h5DataSet), m_imageWidth(imageWidth),
    m_imageHeight(imageHeight), m_scanWidth(scanWidth),
    m_scanHeight(scanHeight), m_imageNumInBlock(blockSize),
    m_blockNumInFile(blockNumInFile), m_totalImageNum(totalImageNum),
    m_imageNumbers(imageNumbers)
{}

Block H5Reader::read()
{

  // get to the end of the file, return empty Block
  if (m_currIndex >= m_totalImageNum) {
    std::cout << "data loading finish with index " << m_currIndex << std::endl;
    return Block();
  }

  // get the data pointer from the py object
  auto getItems = m_h5dataset.attr("__getitem__");
  uint32_t upperBound = m_currIndex + m_imageNumInBlock;

  if (upperBound > m_totalImageNum) {
    upperBound = m_totalImageNum;
  }

  auto sliceIndex = py::slice(m_currIndex, upperBound, 1);
  auto part = getItems(sliceIndex);

  py::array pyarray = py::array(part);
  py::buffer_info buf = pyarray.request();
  uint32_t arraySize = pyarray.size();

  uint16_t* ptr = (uint16_t*)buf.ptr;

  Header h = Header(m_imageWidth, m_imageHeight, m_imageNumInBlock,
                    m_blockNumInFile, m_scanWidth, m_scanHeight, m_currIndex, m_imageNumbers);
  Block b = Block(h);

  std::copy(ptr, ptr + arraySize, b.data.get());

  m_currIndex = upperBound;

  return b;
}

H5Reader::iterator H5Reader::begin()
{
  // read data at first time
  return iterator(this);
}

H5Reader::iterator H5Reader::end()
{
  return iterator(nullptr);
}

} // namespace stempy
